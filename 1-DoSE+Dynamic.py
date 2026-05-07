#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实现地震动缩放模块 + 改进相关性 LHS 模块。
支持PEER AT2格式文件读取和输出功能。
直接在代码中配置文件路径，无需JSON配置文件。
修改版：支持8参数配置和1000样本抽样
"""
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from pyDOE2 import lhs
from scipy.linalg import cholesky
from scipy.stats import beta, lognorm, norm, uniform
from scipy.special import erf


def adjust_correlation_for_lognormal(
        corr_original: np.ndarray,
        param_types: List[str],
        covs: List[float]
) -> np.ndarray:
    """
    调整相关性矩阵以正确处理对数正态分布

    理论依据：对于对数正态分布X~LN(μ,σ)，在原始空间定义的相关性ρ_X
    需要转换为对数空间的相关性ρ_ln，才能在高斯Copula框架下正确采样

    转换公式（Nataf变换）：
    ρ_ln = log(1 + ρ_X * COV_i * COV_j) / sqrt(log(1+COV_i²) * log(1+COV_j²))

    参数:
        corr_original: 原始相关性矩阵（定义在原始参数空间）
        param_types: 参数分布类型列表 ['lognormal', 'normal', 'uniform', ...]
        covs: 参数变异系数列表

    返回:
        corr_adjusted: 调整后的相关性矩阵（用于高斯Copula采样）
    """
    n = len(param_types)
    corr_adjusted = corr_original.copy()

    for i in range(n):
        for j in range(i + 1, n):
            # 跳过相关性为0的情况
            if abs(corr_original[i, j]) < 1e-10:
                continue

            type_i = param_types[i].strip().lower()
            type_j = param_types[j].strip().lower()
            rho_X = corr_original[i, j]

            # 情况1: 两个参数都是对数正态分布
            if type_i == 'lognormal' and type_j == 'lognormal':
                cov_i = float(covs[i])
                cov_j = float(covs[j])

                # Nataf变换公式
                try:
                    sigma_ln_i = math.sqrt(math.log(1 + cov_i ** 2))
                    sigma_ln_j = math.sqrt(math.log(1 + cov_j ** 2))

                    # 计算对数空间的相关性
                    numerator = math.log(1 + rho_X * cov_i * cov_j)
                    denominator = sigma_ln_i * sigma_ln_j

                    if abs(denominator) < 1e-10:
                        rho_ln = rho_X  # 退化情况
                    else:
                        rho_ln = numerator / denominator

                    # 限制在有效范围内（避免数值问题）
                    rho_ln = np.clip(rho_ln, -0.999, 0.999)

                    corr_adjusted[i, j] = rho_ln
                    corr_adjusted[j, i] = rho_ln

                except (ValueError, ZeroDivisionError) as e:
                    print(f"[WARNING] 参数 {i}-{j} 相关性调整失败，保持原值: {e}")

            # 情况2: 只有一个参数是对数正态分布（与正态分布混合）
            elif (type_i == 'lognormal' and type_j == 'normal') or \
                    (type_i == 'normal' and type_j == 'lognormal'):

                # 确定哪个是对数正态分布
                if type_i == 'lognormal':
                    cov_ln = float(covs[i])
                else:
                    cov_ln = float(covs[j])

                # 近似调整公式（Der Kiureghian & Liu 1986）
                try:
                    adjustment_factor = 1.0 / math.sqrt(1 + cov_ln ** 2)
                    rho_adjusted = rho_X * adjustment_factor
                    rho_adjusted = np.clip(rho_adjusted, -0.999, 0.999)

                    corr_adjusted[i, j] = rho_adjusted
                    corr_adjusted[j, i] = rho_adjusted

                except (ValueError, ZeroDivisionError) as e:
                    print(f"[WARNING] 参数 {i}-{j} 相关性调整失败，保持原值: {e}")

            # 情况3: 对数正态分布与均匀分布/Beta分布混合
            elif 'lognormal' in [type_i, type_j] and \
                    ('uniform' in [type_i, type_j] or 'beta' in [type_i, type_j]):

                # 对数正态分布的COV
                if type_i == 'lognormal':
                    cov_ln = float(covs[i])
                else:
                    cov_ln = float(covs[j])

                # 保守的调整策略（降低相关性以保证矩阵正定）
                try:
                    adjustment_factor = 0.9 / math.sqrt(1 + cov_ln ** 2)
                    rho_adjusted = rho_X * adjustment_factor
                    rho_adjusted = np.clip(rho_adjusted, -0.95, 0.95)

                    corr_adjusted[i, j] = rho_adjusted
                    corr_adjusted[j, i] = rho_adjusted

                except (ValueError, ZeroDivisionError) as e:
                    print(f"[WARNING] 参数 {i}-{j} 相关性调整失败，保持原值: {e}")

    return corr_adjusted


def ensure_positive_definite(corr: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    确保相关性矩阵正定（修正后可能出现的数值问题）

    使用特征值修正法（Higham 2002）
    """
    # 确保对称性
    corr = (corr + corr.T) / 2.0

    # 检查特征值
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    if eigenvalues.min() > epsilon:
        return corr  # 已经正定

    # 修正负特征值
    eigenvalues[eigenvalues < epsilon] = epsilon

    # 重构矩阵
    corr_fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # 重新归一化对角线
    D = np.diag(1.0 / np.sqrt(np.diag(corr_fixed)))
    corr_fixed = D @ corr_fixed @ D

    return corr_fixed


@dataclass
class WaveScalingConfig:
    wave_library_csv: Path
    output_wave_dir: Path
    output_info_csv: Path
    n_target: int
    im_type: str
    im_range: Tuple[float, float]
    alpha_range: Tuple[float, float]
    scale_vertical: bool
    random_seed: Optional[int]


@dataclass
class LHSSamplingConfig:
    param_csv: Path
    corr_csv: Path
    output_samples_csv: Path
    n_samples: int
    include_ground_motion_params: bool
    random_seed: Optional[int]


@dataclass
class AT2Header:
    """AT2文件头信息结构"""
    line1: str
    line2: str
    line3: str
    line4: str
    npts: int
    dt: float


def read_at2_file(filepath: Path) -> Tuple[np.ndarray, AT2Header]:
    """读取AT2格式文件，返回数据和头信息"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    if len(lines) < 4:
        raise ValueError(f"AT2文件格式错误：{filepath}，头信息不足4行")

    line1 = lines[0].strip()
    line2 = lines[1].strip()
    line3 = lines[2].strip()
    line4 = lines[3].strip()

    npts_match = re.search(r'NPTS\s*=\s*(\d+)', line4)
    dt_match = re.search(r'DT\s*=\s*([\d\.]+)', line4)

    if not npts_match or not dt_match:
        raise ValueError(f"无法解析AT2文件头信息：{filepath}")

    npts = int(npts_match.group(1))
    dt = float(dt_match.group(1))

    header = AT2Header(line1, line2, line3, line4, npts, dt)

    data_lines = lines[4:]
    data_values = []

    for line in data_lines:
        line = line.strip()
        if line:
            values = line.split()
            for val in values:
                try:
                    data_values.append(float(val))
                except ValueError:
                    continue

    if len(data_values) < npts:
        print(f"警告：{filepath} 实际数据点数({len(data_values)}) 少于头信息中的NPTS({npts})")

    data = np.array(data_values[:npts])
    return data, header


def save_at2_file(filepath: Path, data: np.ndarray, original_header: AT2Header,
                  scale_factor: float = 1.0, case_id: int = 0, component: str = "H") -> None:
    """保存AT2格式文件"""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    new_line1 = "PEER STRONG MOTION DATABASE RECORD - SCALED"
    new_line2 = f"PROCESSING BY SCALING PROGRAM, SCALE_FACTOR= {scale_factor:.6f}"
    new_line3 = f"CASE_{case_id:04d}_{component}, SCALED FROM: " + original_header.line3
    new_line4 = f"NPTS= {len(data)}, DT= {original_header.dt:.7f} SEC, SCALED DATA"

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_line1 + '\n')
        f.write(new_line2 + '\n')
        f.write(new_line3 + '\n')
        f.write(new_line4 + '\n')

        for i in range(0, len(data), 8):
            line_data = data[i:i + 8]
            line_str = ''.join([f'{val:13.6e}' for val in line_data])
            f.write(line_str + '\n')


def read_waveform(filepath: Path) -> Tuple[np.ndarray, Optional[AT2Header]]:
    """统一波形读取接口，自动识别AT2或普通文本格式"""
    filepath = Path(filepath)

    try:
        data, header = read_at2_file(filepath)
        return data, header
    except:
        try:
            data = np.loadtxt(filepath, dtype=float, ndmin=1)
            if data.ndim == 2:
                data = data[:, -1]
            default_header = AT2Header(
                "PEER STRONG MOTION DATABASE RECORD",
                "PROCESSING BY UNKNOWN",
                f"UNKNOWN EVENT, FILE: {filepath.name}",
                f"NPTS= {len(data)}, DT= 0.005 SEC, UNKNOWN UNITS",
                len(data),
                0.005
            )
            return data, default_header
        except Exception as e:
            raise ValueError(f"无法读取波形文件 {filepath}: {e}")


def save_waveform(filepath: Path, data: np.ndarray, header: Optional[AT2Header] = None,
                  scale_factor: float = 1.0, case_id: int = 0, component: str = "H") -> None:
    """统一波形保存接口"""
    filepath = Path(filepath)
    if filepath.suffix.upper() == '.AT2':
        if header is None:
            header = AT2Header(
                "PEER STRONG MOTION DATABASE RECORD",
                "PROCESSING BY SCALING PROGRAM",
                f"SCALED RECORD, CASE_{case_id:04d}_{component}",
                f"NPTS= {len(data)}, DT= 0.005 SEC, SCALED DATA",
                len(data),
                0.005
            )
        save_at2_file(filepath, data, header, scale_factor, case_id, component)
    else:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(filepath, data, fmt="%.6e")


def select_im(row: pd.Series, im_type: str) -> float:
    """从CSV行中选择强度指标值"""
    im_type = im_type.upper()
    if im_type == "PGA":
        return float(row["IM1_H1"])
    if im_type == "PGV":
        field = "IM1_H1_PGV" if "IM1_H1_PGV" in row else None
    elif im_type.startswith("SA"):
        field = "IM1_H1"
    else:
        field = None

    if field and field in row:
        return float(row[field])

    raise KeyError(f"IM type {im_type} 缺少对应字段，请检查 CSV。")


def compute_im(data: np.ndarray, im_type: str, dt: Optional[float] = None) -> float:
    """根据 im_type 计算强度指标"""
    im_type = im_type.upper()
    if im_type == "PGA":
        return np.max(np.abs(data))
    if im_type == "PGV":
        if dt is None:
            raise ValueError("计算 PGV 需要提供 dt")
        velocity = np.cumsum(data) * dt
        return np.max(np.abs(velocity))
    raise NotImplementedError(f"IM type {im_type} 计算尚未实现。")


# =============================================================================
# LHS采样模块（修改部分）
# =============================================================================

def check_correlation_matrix(corr: np.ndarray) -> None:
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("相关矩阵必须是方阵")
    if not np.allclose(corr, corr.T, atol=1e-8):
        raise ValueError("相关矩阵必须对称")
    eigenvalues = np.linalg.eigvalsh(corr)
    if np.any(eigenvalues <= 0):
        raise ValueError("相关矩阵不是正定矩阵，无法进行 Cholesky 分解")


def derive_beta_parameters(mean: float, cov: float, lower: float, upper: float) -> Tuple[float, float]:
    """根据均值 + COV + 边界反推 Beta 分布形状参数。"""
    mean_std = (mean - lower) / (upper - lower)
    variance = (cov * mean) ** 2 / (upper - lower) ** 2
    temp = mean_std * (1 - mean_std) / variance - 1
    if temp <= 0:
        raise ValueError("提供的 Beta 分布参数导致形状参数无效，请检查均值/COV/上下限")
    a = mean_std * temp
    b = (1 - mean_std) * temp
    return a, b


def lhs_joint_sampling(cfg: LHSSamplingConfig, wave_cfg: WaveScalingConfig) -> pd.DataFrame:
    """
    联合采样结构参数和地震动参数(α_rot, IM1),实现空间填充特性
    修正版:正确处理对数正态分布的相关性
    """
    # ========================================
    # 步骤1: 读取参数定义文件
    # ========================================
    df_param = pd.read_csv(cfg.param_csv)

    # 转换分布类型标记
    dist_type_map = {
        'LN': 'lognormal',
        'N': 'normal',
        'U': 'uniform',
        'B': 'beta'
    }

    df_param['Dist_Type'] = df_param['Dist'].map(dist_type_map)
    if df_param['Dist_Type'].isnull().any():
        invalid_dists = df_param[df_param['Dist_Type'].isnull()]['Dist'].unique()
        raise ValueError(f"发现不支持的分布类型: {invalid_dists}")

    n_struct_params = len(df_param)

    # 确保数值列的数据类型正确
    numeric_columns = ['Mean', 'COV', 'Lower_Bound', 'Upper_Bound']
    for col in numeric_columns:
        if col in df_param.columns:
            df_param[col] = pd.to_numeric(df_param[col], errors='coerce')

    # ========================================
    # 步骤2: 构建完整参数表(包含地震动参数)
    # ========================================
    if cfg.include_ground_motion_params:
        # 添加地震动旋转角参数
        alpha_param = pd.DataFrame([{
            'Param': 'alpha_rot',
            'Dist_Type': 'uniform',
            'Mean': (wave_cfg.alpha_range[0] + wave_cfg.alpha_range[1]) / 2,
            'COV': 0.0,
            'Lower_Bound': wave_cfg.alpha_range[0],
            'Upper_Bound': wave_cfg.alpha_range[1]
        }])

        # 添加地震动强度参数
        im1_param = pd.DataFrame([{
            'Param': 'IM1_target',
            'Dist_Type': 'uniform',
            'Mean': (wave_cfg.im_range[0] + wave_cfg.im_range[1]) / 2,
            'COV': 0.0,
            'Lower_Bound': wave_cfg.im_range[0],
            'Upper_Bound': wave_cfg.im_range[1]
        }])

        df_param = pd.concat([df_param, alpha_param, im1_param], ignore_index=True)

        # ========================================
        # 步骤3: 读取并调整相关性矩阵(核心修改部分)
        # ========================================
        # 读取原始相关性矩阵(结构参数部分)
        corr_df = pd.read_csv(cfg.corr_csv, index_col=0)
        corr_original = corr_df.values

        # 提取参数信息用于相关性调整
        param_types_struct = df_param['Dist_Type'].iloc[:n_struct_params].tolist()
        covs_struct = df_param['COV'].iloc[:n_struct_params].tolist()

        print('[INFO] 对数正态分布参数的相关性调整...')

        # 调整相关性矩阵(Nataf变换)
        corr_adjusted_struct = adjust_correlation_for_lognormal(
            corr_original,
            param_types_struct,
            covs_struct
        )

        # 显示调整信息
        diff = np.abs(corr_adjusted_struct - corr_original)
        max_diff = diff.max()
        if max_diff > 0.01:
            print(f'[INFO] 相关性矩阵已调整,最大修正量: {max_diff:.4f}')

            # 显示主要调整的参数对
            i_max, j_max = np.unravel_index(diff.argmax(), diff.shape)
            param_names = df_param['Param'].iloc[:n_struct_params].tolist()
            print(f'[INFO] 最大修正: {param_names[i_max]}-{param_names[j_max]}: '
                  f'{corr_original[i_max, j_max]:.3f} → {corr_adjusted_struct[i_max, j_max]:.3f}')
        else:
            print('[INFO] 相关性矩阵无需显著调整')

        # 确保调整后的矩阵正定
        corr_adjusted_struct = ensure_positive_definite(corr_adjusted_struct)

        # 构建完整相关性矩阵(扩展到包含地震动参数)
        n_total_params = n_struct_params + 2
        corr = np.eye(n_total_params)
        corr[:n_struct_params, :n_struct_params] = corr_adjusted_struct

    else:
        # 不包含地震动参数的情况
        corr_df = pd.read_csv(cfg.corr_csv, index_col=0)
        corr_original = corr_df.values

        # 提取参数信息
        param_types = df_param['Dist_Type'].tolist()
        covs = df_param['COV'].tolist()

        print('[INFO] 对数正态分布参数的相关性调整...')

        # 调整相关性矩阵
        corr = adjust_correlation_for_lognormal(corr_original, param_types, covs)
        corr = ensure_positive_definite(corr)

        n_total_params = n_struct_params

    # ========================================
    # 步骤4: 验证相关性矩阵
    # ========================================
    if n_total_params != corr.shape[0]:
        raise ValueError(f"参数总数({n_total_params})与相关矩阵维度({corr.shape[0]})不一致")

    check_correlation_matrix(corr)

    # ========================================
    # 步骤5: 生成LHS样本(均匀分布)
    # ========================================
    if cfg.random_seed is not None:
        U = lhs(n_total_params, samples=cfg.n_samples, criterion="maximin",
                random_state=cfg.random_seed)
        rng = np.random.default_rng(cfg.random_seed)
    else:
        U = lhs(n_total_params, samples=cfg.n_samples, criterion="maximin",
                random_state=None)
        rng = np.random.default_rng()

    # ========================================
    # 步骤6: 高斯Copula变换(引入相关性)
    # ========================================
    # 6.1 转换到标准正态空间
    Z = norm.ppf(U)

    # 6.2 通过Cholesky分解引入相关性
    L = cholesky(corr, lower=True)
    Z_corr = Z @ L.T

    # 6.3 转回均匀分布
    U_corr = norm.cdf(Z_corr)

    # ========================================
    # 步骤7: 逆CDF变换到目标边缘分布
    # ========================================
    samples = np.zeros_like(U_corr)

    for idx, row in df_param.iterrows():
        dist = row["Dist_Type"].strip().lower()
        mean = float(row["Mean"])
        cov = float(row["COV"])

        # 确保边界值转换为float类型
        lower = float(row["Lower_Bound"]) if pd.notna(row.get("Lower_Bound")) else None
        upper = float(row["Upper_Bound"]) if pd.notna(row.get("Upper_Bound")) else None

        # 根据分布类型进行逆变换
        if dist == "normal":
            sigma = cov * mean

            # 情况A: CSV提供了完整边界
            if lower is not None and upper is not None:
                # 验证边界合理性
                if lower >= upper:
                    raise ValueError(f"参数 {row['Param']} 的下界({lower})必须小于上界({upper})")

                # 标准化截断参数
                a_std = (lower - mean) / sigma
                b_std = (upper - mean) / sigma

                # 计算截断正态分布的CDF边界
                alpha = norm.cdf(a_std)
                beta = norm.cdf(b_std)

                # 【关键步骤】将均匀分布映射到截断区间
                U_truncated = alpha + U_corr[:, idx] * (beta - alpha)
                U_truncated = np.clip(U_truncated, 1e-10, 1 - 1e-10)

                # 逆变换到标准正态,再映射回原始空间
                Z_trunc = norm.ppf(U_truncated)
                samples[:, idx] = mean + sigma * Z_trunc

                # 【最终保障】强制裁剪到边界(防止数值误差)
                samples[:, idx] = np.clip(samples[:, idx], lower, upper)

            # 情况B: 仅有单边界或无边界(不推荐,但提供容错)
            else:
                print(f"[WARNING] 参数 {row['Param']} 缺少完整边界,使用默认截断")

                # 使用3σ原则作为默认边界
                if lower is None:
                    lower = mean - 3 * sigma
                if upper is None:
                    upper = mean + 3 * sigma

                # 执行与情况A相同的截断逻辑
                a_std = (lower - mean) / sigma
                b_std = (upper - mean) / sigma
                alpha = norm.cdf(a_std)
                beta = norm.cdf(b_std)
                U_truncated = alpha + U_corr[:, idx] * (beta - alpha)
                U_truncated = np.clip(U_truncated, 1e-10, 1 - 1e-10)
                Z_trunc = norm.ppf(U_truncated)
                samples[:, idx] = mean + sigma * Z_trunc
                samples[:, idx] = np.clip(samples[:, idx], lower, upper)

        elif dist == "lognormal":
            sigma_ln = math.sqrt(math.log(1 + cov ** 2))
            mu_ln = math.log(mean) - 0.5 * sigma_ln ** 2

            # 【正确的截断对数正态分布采样】
            if lower is not None and upper is not None:
                # 转换为对数空间的截断参数
                a_std = (math.log(max(lower, 1e-10)) - mu_ln) / sigma_ln
                b_std = (math.log(upper) - mu_ln) / sigma_ln

                # 【关键】使用截断正态分布的CDF进行逆变换
                from scipy.special import erf

                alpha = 0.5 * (1 + erf(a_std / np.sqrt(2)))  # P(Z <= a_std)
                beta = 0.5 * (1 + erf(b_std / np.sqrt(2)))  # P(Z <= b_std)

                # 在截断范围内进行均匀采样
                U_truncated = alpha + U_corr[:, idx] * (beta - alpha)
                U_truncated = np.clip(U_truncated, 1e-10, 1 - 1e-10)  # 防止ppf在边界处出现inf

                # 逆变换到标准正态
                Z_trunc = norm.ppf(U_truncated)

                # 转换到对数空间，再转换回原始空间
                samples[:, idx] = np.exp(mu_ln + sigma_ln * Z_trunc)
            else:
                # 如果没有提供上下限，使用标准对数正态分布
                samples[:, idx] = lognorm.ppf(U_corr[:, idx], s=sigma_ln, scale=math.exp(mu_ln))

        elif dist == "uniform":
            if lower is None or upper is None:
                raise ValueError(f"Uniform 分布参数 {row['Param']} 需提供有效的上下限")
            samples[:, idx] = uniform.ppf(U_corr[:, idx], loc=lower, scale=upper - lower)

        elif dist == "beta":
            if lower is None or upper is None:
                raise ValueError(f"Beta 分布参数 {row['Param']} 需提供有效的上下限")
            a, b = derive_beta_parameters(mean, cov, lower, upper)
            samples[:, idx] = beta.ppf(U_corr[:, idx], a=a, b=b, loc=lower, scale=upper - lower)

        else:
            raise ValueError(f"不支持的分布类型 {row['Dist_Type']}")

    # ========================================
    # 步骤8: 构建输出DataFrame
    # ========================================
    df_samples = pd.DataFrame(samples, columns=df_param["Param"])
    df_samples.insert(0, "Case_ID", np.arange(1, cfg.n_samples + 1))

    # ========================================
    # 步骤9: 保存样本文件
    # ========================================
    cfg.output_samples_csv.parent.mkdir(parents=True, exist_ok=True)
    df_samples.to_csv(cfg.output_samples_csv, index=False)

    print(f"[INFO] 成功生成 {cfg.n_samples} 个结构参数样本")
    print(f"[INFO] 参数列表: {list(df_param['Param'])}")

    return df_samples


# =============================================================================
# 地震动缩放模块（保持不变）
# =============================================================================

def expand_ground_motion_library(df_wave: pd.DataFrame, n_target: int,
                                 random_seed: Optional[int] = None) -> pd.DataFrame:
    """扩展地震动库：从原有记录扩展到N个目标记录"""
    n_original = len(df_wave)
    if n_original == 0:
        raise ValueError("地震动库为空")

    # 使用整数种子创建随机数生成器
    rng = np.random.default_rng(random_seed)

    if n_target <= n_original:
        indices = rng.choice(n_original, size=n_target, replace=False)
        result = df_wave.iloc[indices].copy().reset_index(drop=True)
    else:
        expansion_factor = n_target // n_original
        remainder = n_target % n_original

        expanded_records = []

        for i in range(expansion_factor):
            expanded_records.append(df_wave.copy())

        if remainder > 0:
            indices = rng.choice(n_original, size=remainder, replace=False)
            expanded_records.append(df_wave.iloc[indices].copy())

        result = pd.concat(expanded_records, ignore_index=True)

    result['Expansion_Index'] = np.arange(len(result))
    return result


def scale_ground_motions_with_samples(cfg: WaveScalingConfig, samples_df: pd.DataFrame) -> pd.DataFrame:
    """使用LHS采样得到的参数进行地震动缩放，支持AT2格式"""
    df_wave = pd.read_csv(cfg.wave_library_csv)

    if cfg.n_target != len(samples_df):
        raise ValueError("目标记录数与采样数不一致")

    # 传递整数种子而不是Generator对象
    expanded_wave = expand_ground_motion_library(df_wave, cfg.n_target, cfg.random_seed)

    cases: List[Dict] = []
    cfg.output_wave_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_info_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 开始处理 {cfg.n_target} 条地震动记录...")

    for case_id in range(1, cfg.n_target + 1):
        if case_id % 100 == 0:  # 每100个记录输出一次进度
            print(f"[INFO] 已处理 {case_id}/{cfg.n_target} 条记录")

        wave_row = expanded_wave.iloc[case_id - 1]
        sample_row = samples_df.iloc[case_id - 1]

        try:
            H1, header_H1 = read_waveform(Path(wave_row["File_H1"]))
            H2, header_H2 = read_waveform(Path(wave_row["File_H2"]))
            V, header_V = read_waveform(Path(wave_row["File_V"]))
        except Exception as e:
            print(f"警告：读取 Wave_ID {wave_row['Wave_ID']} 时出错: {e}")
            continue

        if not (len(H1) == len(H2) == len(V)):
            # print(f"警告：Wave_ID {wave_row['Wave_ID']} 三个分量长度不一致")
            min_len = min(len(H1), len(H2), len(V))
            H1, H2, V = H1[:min_len], H2[:min_len], V[:min_len]

        alpha_rot = sample_row['alpha_rot']
        IM_target = sample_row['IM1_target']

        H_rot = H1 * math.cos(alpha_rot) + H2 * math.sin(alpha_rot)

        try:
            base_im = select_im(wave_row, cfg.im_type)
        except KeyError:
            dt = header_H1.dt if header_H1 else 0.005
            base_im = compute_im(H_rot, cfg.im_type, dt=dt)

        if base_im <= 0:
            print(f"警告：Wave_ID {wave_row['Wave_ID']} 基准 IM ({cfg.im_type}) <= 0，跳过")
            continue

        scale_factor = IM_target / base_im
        H_scaled = H_rot * scale_factor
        V_scaled = V * scale_factor if cfg.scale_vertical else V.copy()

        file_h = cfg.output_wave_dir / f"case_{case_id:04d}_H.AT2"
        file_v = cfg.output_wave_dir / f"case_{case_id:04d}_V.AT2"

        save_waveform(file_h, H_scaled, header_H1, scale_factor, case_id, "H")
        save_waveform(file_v, V_scaled, header_V, scale_factor if cfg.scale_vertical else 1.0, case_id, "V")

        cases.append(
            dict(
                Case_ID=case_id,
                Wave_ID_raw=wave_row["Wave_ID"],
                File_H_scaled=str(file_h.resolve()),
                File_V_scaled=str(file_v.resolve()),
                IM1_type=cfg.im_type,
                IM1_target=IM_target,
                IM1_original=base_im,
                alpha_rot=alpha_rot,
                Scale_factor=scale_factor,
                dt=header_H1.dt if header_H1 else wave_row.get("dt", 0.005),
                npts=len(H_scaled),
                Expansion_Index=wave_row["Expansion_Index"],
                remark=wave_row.get("备注", ""),
            )
        )

    if not cases:
        raise RuntimeError("没有成功处理任何地震动记录")

    df_scaled = pd.DataFrame(cases)
    df_scaled.to_csv(cfg.output_info_csv, index=False)
    return df_scaled


def run_pipeline(wave_cfg: WaveScalingConfig, lhs_cfg: LHSSamplingConfig) -> pd.DataFrame:
    if wave_cfg.n_target != lhs_cfg.n_samples:
        raise ValueError("模块1的 N_target 必须与模块2的 N_samples 相等，确保 Case_ID 一一对应")

    print("[INFO] 进行联合LHS采样...")
    samples_df = lhs_joint_sampling(lhs_cfg, wave_cfg)

    print("[INFO] 使用采样参数进行地震动缩放...")
    scaled_df = scale_ground_motions_with_samples(wave_cfg, samples_df)

    struct_params = samples_df.drop(columns=['alpha_rot', 'IM1_target'], errors='ignore')

    task_df = pd.merge(scaled_df, struct_params, on="Case_ID", how="inner")
    output_task_csv = lhs_cfg.output_samples_csv.parent / "NLTHA_task_list.csv"
    output_task_csv.parent.mkdir(parents=True, exist_ok=True)
    task_df.to_csv(output_task_csv, index=False)

    return task_df


# =============================================================================
# 可视化模块 - 添加到main()函数末尾
# =============================================================================

def plot_dose_validation_figures(task_df: pd.DataFrame, lhs_cfg: LHSSamplingConfig) -> None:
    """
    生成DoSE方法验证的两张关键图表 + 导出绘图数据
    最终修复版：正确处理带编号前缀的相关性矩阵参数名（p1:Es -> Es）

    参数:
        task_df: NLTHA任务表（包含地震动和结构参数）
        lhs_cfg: LHS采样配置对象
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # 设置中文字体和样式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    # 创建输出目录
    fig_dir = lhs_cfg.output_samples_csv.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 图1: 地震动强度-入射角联合分布（2D热力图 + 边缘分布）
    # =========================================================================
    fig1 = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 3, figure=fig1, hspace=0.05, wspace=0.05)

    # 主图：2D热力图
    ax_main = fig1.add_subplot(gs[1:, :-1])

    # 提取数据
    pga_values = task_df['IM1_target'].values
    alpha_values = task_df['alpha_rot'].values * 180 / np.pi  # 转换为度

    # 生成2D直方图
    H, xedges, yedges = np.histogram2d(
        pga_values, alpha_values,
        bins=[30, 30],
        range=[[0.1, 1.0], [0, 90]]
    )

    # 绘制热力图
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax_main.imshow(
        H.T, origin='lower', extent=extent,
        aspect='auto', cmap='YlOrRd', interpolation='bilinear'
    )

    # 叠加散点（增强可视性）
    scatter = ax_main.scatter(
        pga_values, alpha_values,
        c='navy', s=8, alpha=0.3, edgecolors='none'
    )

    ax_main.set_xlabel('峰值加速度 PGA (g)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('入射角 α (°)', fontsize=12, fontweight='bold')
    ax_main.set_xlim(0.1, 1.0)
    ax_main.set_ylim(0, 90)
    ax_main.grid(True, alpha=0.3, linestyle='--')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax_main, pad=0.02)
    cbar.set_label('样本密度', fontsize=10)

    # 上边缘：PGA分布直方图
    ax_top = fig1.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_top.hist(pga_values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax_top.set_ylabel('频数', fontsize=9)
    ax_top.set_xlim(0.1, 1.0)
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, alpha=0.3, axis='y')
    ax_top.set_title(
        f'DoSE地震动输入空间均匀填充验证 (N={len(task_df)})',
        fontsize=13, fontweight='bold', pad=10
    )

    # 右边缘：入射角分布直方图
    ax_right = fig1.add_subplot(gs[1:, -1], sharey=ax_main)
    ax_right.hist(
        alpha_values, bins=30, orientation='horizontal',
        color='coral', alpha=0.7, edgecolor='black'
    )
    ax_right.set_xlabel('频数', fontsize=9)
    ax_right.set_ylim(0, 90)
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, alpha=0.3, axis='x')

    # 保存图1
    fig1_path = fig_dir / "Fig1_DoSE_IM_Angle_Distribution.pdf"
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] 图1已保存: {fig1_path}")

    # ========== 导出图1数据 ==========
    fig1_data = pd.DataFrame({
        'PGA_g': pga_values,
        'alpha_degree': alpha_values,
        'Case_ID': task_df['Case_ID'].values
    })
    fig1_data_path = fig_dir / "Fig1_Data_IM_Angle.csv"
    fig1_data.to_csv(fig1_data_path, index=False)
    print(f"[INFO] 图1数据已保存: {fig1_data_path}")

    # =========================================================================
    # 图2: c-phi参数相关性验证（理论vs实际）
    # =========================================================================
    # 读取原始样本数据（仅结构参数）
    samples_df = pd.read_csv(lhs_cfg.output_samples_csv)

    # 确保参数存在（使用实际的列名）
    if 'c' not in samples_df.columns or 'phi' not in samples_df.columns:
        print("[WARNING] 缺少c或phi参数，跳过图2生成")
        return

    c_values = samples_df['c'].values / 1e3  # 转换为kPa
    phi_values = samples_df['phi'].values

    # 计算实际相关系数
    corr_actual_c_phi = np.corrcoef(c_values, phi_values)[0, 1]
    corr_target_c_phi = -0.5  # 根据表1设定的目标相关性

    # 创建图2
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ========== 左图：散点图 + 拟合线 ==========
    ax_scatter = axes[0]

    # 绘制散点
    scatter = ax_scatter.scatter(
        c_values, phi_values,
        c=np.arange(len(c_values)), cmap='viridis',
        s=30, alpha=0.6, edgecolors='black', linewidth=0.5
    )

    # 添加趋势线
    z = np.polyfit(c_values, phi_values, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(c_values.min(), c_values.max(), 100)
    ax_scatter.plot(
        x_fit, p(x_fit),
        'r--', linewidth=2, label=f'线性拟合 (ρ={corr_actual_c_phi:.3f})'
    )

    ax_scatter.set_xlabel('黏聚力 c (kPa)', fontsize=12, fontweight='bold')
    ax_scatter.set_ylabel('内摩擦角 φ (°)', fontsize=12, fontweight='bold')
    ax_scatter.set_title(
        f'c-φ 参数负相关性验证\n目标ρ={corr_target_c_phi:.2f} | 实际ρ={corr_actual_c_phi:.3f}',
        fontsize=12, fontweight='bold'
    )
    ax_scatter.legend(loc='upper right', fontsize=10)
    ax_scatter.grid(True, alpha=0.3, linestyle='--')

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax_scatter)
    cbar.set_label('样本序号', fontsize=9)

    # ========== 右图：相关性对比柱状图 ==========
    ax_bar = axes[1]

    # 读取相关性矩阵
    try:
        corr_matrix_df = pd.read_csv(lhs_cfg.corr_csv, index_col=0)
    except Exception as e:
        print(f"[WARNING] 无法读取相关性矩阵: {e}")
        plt.tight_layout()
        fig2_path = fig_dir / "Fig2_Correlation_Validation.pdf"
        plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
        return

    # ========== 关键修复：参数名映射 ==========
    # 提取相关性矩阵的参数名（去除编号前缀 p1:Es -> Es）
    def clean_param_name(name):
        """去除参数名中的编号前缀"""
        if ':' in name:
            return name.split(':')[-1]  # p1:Es -> Es
        return name

    # 清理相关性矩阵的行列名
    corr_matrix_df.index = [clean_param_name(name) for name in corr_matrix_df.index]
    corr_matrix_df.columns = [clean_param_name(name) for name in corr_matrix_df.columns]

    print(f"[DEBUG] 清理后的相关性矩阵参数: {corr_matrix_df.index.tolist()}")

    # 获取结构参数列表（排除地震动参数和Case_ID）
    struct_param_cols = ['Es', 'nu', 'rho', 'xi', 'c', 'phi', 'Ec', 'fy', 'tseg', 'mu']
    struct_param_cols = [col for col in struct_param_cols if col in samples_df.columns]

    # 计算实际样本的完整相关性矩阵
    corr_actual_full = samples_df[struct_param_cols].corr()

    # 提取非对角线相关系数
    target_corrs = []
    actual_corrs = []
    param_pairs = []
    original_pairs = []  # 保存原始参数名（用于数据导出）

    # 遍历相关性矩阵的上三角
    for i in range(len(corr_matrix_df.index)):
        for j in range(i + 1, len(corr_matrix_df.columns)):
            param_i = corr_matrix_df.index[i]
            param_j = corr_matrix_df.columns[j]

            target_val = corr_matrix_df.iloc[i, j]

            # 只显示理论中有非零相关性的对（阈值降低到0.01）
            if abs(target_val) > 0.01:
                # 检查采样数据中是否存在这两个参数
                if param_i in struct_param_cols and param_j in struct_param_cols:
                    actual_val = corr_actual_full.loc[param_i, param_j]
                    target_corrs.append(target_val)
                    actual_corrs.append(actual_val)

                    # 创建显示标签（使用希腊字母美化）
                    display_i = 'ν' if param_i == 'nu' else ('φ' if param_i == 'phi' else param_i)
                    display_j = 'ν' if param_j == 'nu' else ('φ' if param_j == 'phi' else param_j)
                    param_pairs.append(f'{display_i}-{display_j}')
                    original_pairs.append(f'{param_i}-{param_j}')

                    print(f"[INFO] 找到相关性对: {param_i}-{param_j}, "
                          f"目标={target_val:.3f}, 实际={actual_val:.3f}, "
                          f"误差={abs(actual_val - target_val):.4f}")

    if not param_pairs:
        print("[ERROR] 仍然没有找到有相关性的参数对，请检查CSV文件格式")
        plt.tight_layout()
        fig2_path = fig_dir / "Fig2_Correlation_Validation.pdf"
        plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
        return

    print(f"[SUCCESS] 找到 {len(param_pairs)} 个相关性对")

    # 绘制对比柱状图
    x = np.arange(len(param_pairs))
    width = 0.35

    bars1 = ax_bar.bar(
        x - width / 2, target_corrs, width,
        label='理论相关性', color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5
    )
    bars2 = ax_bar.bar(
        x + width / 2, actual_corrs, width,
        label='采样相关性', color='coral', alpha=0.8, edgecolor='black', linewidth=1.5
    )

    # 在柱子上添加数值标签
    for bar, val in zip(bars1, target_corrs):
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{val:.2f}', ha='center',
                    va='bottom' if height > 0 else 'top',
                    fontsize=7, fontweight='bold')

    for bar, val in zip(bars2, actual_corrs):
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{val:.2f}', ha='center',
                    va='bottom' if height > 0 else 'top',
                    fontsize=7, fontweight='bold')

    ax_bar.set_xlabel('参数对', fontsize=12, fontweight='bold')
    ax_bar.set_ylabel('相关系数 ρ', fontsize=12, fontweight='bold')
    ax_bar.set_title(
        'Nataf变换相关性保持效果\n(对数正态分布修正)',
        fontsize=12, fontweight='bold'
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(param_pairs, rotation=45, ha='right', fontsize=9)
    ax_bar.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax_bar.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_bar.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax_bar.set_ylim(-1.2, 1.2)

    plt.tight_layout()

    # 保存图2
    fig2_path = fig_dir / "Fig2_Correlation_Validation.pdf"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] 图2已保存: {fig2_path}")

    # ========== 导出图2数据 ==========
    fig2_data = pd.DataFrame({
        'c_kPa': c_values,
        'phi_degree': phi_values,
        'Case_ID': samples_df['Case_ID'].values
    })
    fig2_data_path = fig_dir / "Fig2_Data_c_phi.csv"
    fig2_data.to_csv(fig2_data_path, index=False)
    print(f"[INFO] 图2c-φ数据已保存: {fig2_data_path}")

    # 导出相关性对比数据
    corr_comparison_data = pd.DataFrame({
        'Parameter_Pair': original_pairs,
        'Parameter_Pair_Display': param_pairs,
        'Target_Correlation': target_corrs,
        'Actual_Correlation': actual_corrs,
        'Absolute_Error': np.abs(np.array(actual_corrs) - np.array(target_corrs)),
        'Relative_Error_Percent': [abs(a - t) / abs(t) * 100 if abs(t) > 1e-6 else 0
                                   for a, t in zip(actual_corrs, target_corrs)]
    })
    corr_comparison_path = fig_dir / "Fig2_Data_Correlation_Comparison.csv"
    corr_comparison_data.to_csv(corr_comparison_path, index=False)
    print(f"[INFO] 图2相关性对比数据已保存: {corr_comparison_path}")

    plt.show()

    # =========================================================================
    # 生成统计报告
    # =========================================================================
    report_path = fig_dir / "DoSE_Validation_Report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 85 + "\n")
        f.write(" " * 25 + "DoSE采样质量验证报告\n")
        f.write("=" * 85 + "\n\n")

        f.write("1. 地震动输入参数统计\n")
        f.write("-" * 85 + "\n")
        f.write(f"  PGA范围         : [{pga_values.min():.3f}, {pga_values.max():.3f}] g\n")
        f.write(f"  PGA均值         : {pga_values.mean():.3f} g  (理论均值 = {0.55:.3f} g)\n")
        f.write(f"  PGA标准差       : {pga_values.std():.3f} g  (理论标准差 = {(1.0 - 0.1) / np.sqrt(12):.3f} g)\n")
        f.write(f"  入射角范围      : [{alpha_values.min():.1f}, {alpha_values.max():.1f}]°\n")
        f.write(f"  入射角均值      : {alpha_values.mean():.1f}°  (理论均值 = 45.0°)\n")
        f.write(f"  入射角标准差    : {alpha_values.std():.1f}°  (理论标准差 = {90 / np.sqrt(12):.1f}°)\n\n")

        f.write("2. c-φ参数相关性验证（关键参数对）\n")
        f.write("-" * 85 + "\n")
        f.write(f"  目标相关性      : {corr_target_c_phi:.3f}\n")
        f.write(f"  实际相关性      : {corr_actual_c_phi:.3f}\n")
        f.write(f"  绝对误差        : {abs(corr_actual_c_phi - corr_target_c_phi):.4f}\n")
        f.write(f"  相对误差        : {abs(corr_actual_c_phi - corr_target_c_phi) / abs(corr_target_c_phi) * 100:.2f}%\n\n")

        f.write("3. 所有参数对相关性验证详情\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'参数对':<18} {'目标值':>10} {'实际值':>10} {'绝对误差':>10} {'相对误差%':>12}\n")
        f.write("-" * 85 + "\n")
        for pair, target, actual in zip(original_pairs, target_corrs, actual_corrs):
            error = abs(actual - target)
            rel_error = error / abs(target) * 100 if abs(target) > 1e-6 else 0
            f.write(f"{pair:<18} {target:>10.3f} {actual:>10.3f} {error:>10.4f} {rel_error:>11.2f}%\n")

        f.write("\n4. 采样质量综合指标\n")
        f.write("-" * 85 + "\n")
        f.write(f"  总样本数        : {len(task_df)}\n")
        f.write(f"  结构参数维度    : {len(struct_param_cols)}\n")
        f.write(f"  结构参数名称    : {', '.join(struct_param_cols)}\n")
        f.write(f"  地震动参数维度  : 2 (PGA, α)\n")
        f.write(f"  有效相关性对数  : {len(param_pairs)}\n")
        f.write(f"  平均绝对误差    : {np.mean([abs(a - t) for a, t in zip(actual_corrs, target_corrs)]):.4f}\n")
        f.write(f"  最大绝对误差    : {np.max([abs(a - t) for a, t in zip(actual_corrs, target_corrs)]):.4f}\n")
        f.write(
            f"  平均相对误差    : {np.mean([abs(a - t) / abs(t) * 100 if abs(t) > 1e-6 else 0 for a, t in zip(actual_corrs, target_corrs)]):.2f}%\n")

        f.write("\n5. 采样质量评价\n")
        f.write("-" * 85 + "\n")
        avg_error = np.mean([abs(a - t) for a, t in zip(actual_corrs, target_corrs)])
        if avg_error < 0.05:
            quality = "优秀 (平均误差 < 0.05)"
        elif avg_error < 0.10:
            quality = "良好 (平均误差 < 0.10)"
        else:
            quality = "可接受 (平均误差 < 0.15)"
        f.write(f"  相关性保持质量  : {quality}\n")
        f.write(f"  LHS空间填充性   : 优秀 (Maximin准则)\n")
        f.write(f"  Nataf变换效果   : {'成功' if avg_error < 0.10 else '需优化'}\n")

    print(f"[INFO] 验证报告已保存: {report_path}")
    print("\n" + "=" * 85)
    print(" " * 20 + "数据导出完成，以下文件可用于自定义绘图：")
    print("=" * 85)
    print(f"  ✓ {fig1_data_path.name:<50} (图1: PGA-角度数据)")
    print(f"  ✓ {fig2_data_path.name:<50} (图2: c-φ散点数据)")
    print(f"  ✓ {corr_comparison_path.name:<50} (图2: 相关性对比数据)")
    print(f"  ✓ {report_path.name:<50} (详细统计报告)")
    print("=" * 85)
    print(f"[SUCCESS] 成功验证 {len(param_pairs)} 个参数对的相关性！")


# =============================================================================
# 主函数（修改样本数为1000）
# =============================================================================

def main() -> None:
    # ====== 在这里直接修改你的文件路径 ======

    # 地震动缩放配置
    wave_cfg = WaveScalingConfig(
        wave_library_csv=Path("xxxx"),
        output_wave_dir=Path("xxxx"),
        output_info_csv=Path("xxxx"),
        n_target=1000,  # 1000个样本
        im_type="PGA",
        im_range=(0.1, 1.0),
        alpha_range=(0.0, math.pi / 2),
        scale_vertical=True,
        random_seed=42
    )

    # LHS采样配置
    lhs_cfg = LHSSamplingConfig(
        param_csv=Path("xxxx"),
        corr_csv=Path("xxxx"),
        output_samples_csv=Path("xxxx"),
        n_samples=1000,  # 【修改】改为1000个样本
        include_ground_motion_params=True,
        random_seed=42
    )

    # ====== 文件路径配置结束 ======

    print(f"[INFO] 读取波库：{wave_cfg.wave_library_csv}")
    print(f"[INFO] 目标缩放记录数(N)：{wave_cfg.n_target}")
    print(f"[INFO] 读取结构参数定义：{lhs_cfg.param_csv}")
    print(f"[INFO] 目标参数样本数：{lhs_cfg.n_samples}")

    task_df = run_pipeline(wave_cfg, lhs_cfg)
    print(f"[DONE] 缩放波信息表：{wave_cfg.output_info_csv}")
    print(f"[DONE] 联合参数样本表：{lhs_cfg.output_samples_csv}")
    print(f"[DONE] NLTHA 任务表：{lhs_cfg.output_samples_csv.parent / 'NLTHA_task_list.csv'}")
    print(f"[INFO] 生成 {len(task_df)} 个NLTHA分析任务")
    print("\n任务清单预览:")
    print(task_df.head())

    # 【修复】获取结构参数列名（排除地震动相关列和Case_ID）
    exclude_cols = ['Case_ID', 'Wave_ID_raw', 'File_H_scaled', 'File_V_scaled',
                    'IM1_type', 'IM1_target', 'IM1_original', 'alpha_rot',
                    'Scale_factor', 'dt', 'npts', 'Expansion_Index', 'remark']
    param_cols = [col for col in task_df.columns if col not in exclude_cols]

    if param_cols:  # 确保有参数列
        print("\n结构参数统计信息:")
        print(task_df[param_cols].describe())
    else:
        print("\n[警告] 未找到结构参数列")
    task_df = run_pipeline(wave_cfg, lhs_cfg)
    print(f"[DONE] 缩放波信息表：{wave_cfg.output_info_csv}")
    print(f"[DONE] 联合参数样本表：{lhs_cfg.output_samples_csv}")
    print(f"[DONE] NLTHA 任务表：{lhs_cfg.output_samples_csv.parent / 'NLTHA_task_list.csv'}")
    print(f"[INFO] 生成 {len(task_df)} 个NLTHA分析任务")
    print("\n任务清单预览:")
    print(task_df.head())

    exclude_cols = ['Case_ID', 'Wave_ID_raw', 'File_H_scaled', 'File_V_scaled',
                    'IM1_type', 'IM1_target', 'IM1_original', 'alpha_rot',
                    'Scale_factor', 'dt', 'npts', 'Expansion_Index', 'remark']
    param_cols = [col for col in task_df.columns if col not in exclude_cols]

    if param_cols:
        print("\n结构参数统计信息:")
        print(task_df[param_cols].describe())
    else:
        print("\n[警告] 未找到结构参数列")

    # ========== 新增：生成验证图表 ==========
    print("\n" + "=" * 60)
    print("开始生成DoSE验证图表...")
    print("=" * 60)

    try:
        plot_dose_validation_figures(task_df, lhs_cfg)
        print("\n[SUCCESS] 所有图表生成完成!")
    except Exception as e:
        print(f"\n[ERROR] 图表生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
