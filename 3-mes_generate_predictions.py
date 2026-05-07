# mes_generate_predictions.py
"""
使用训练好的MES模型对新数据进行批量预测
生成10000个样本的预测结果(均值、标准差、置信区间)
"""

import os
import numpy as np
import pandas as pd
import torch
import joblib
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. 配置类 (简化版)
# =============================================================================

class PredictionConfig:
    """预测配置"""
    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径配置 - 根据你的实际路径修改
    MODEL_PATH = "F:/Abaqus_Mu/MSE/results_2/best_mes_model.pt"  # 训练好的模型
    INPUT_CSV = "F:/Abaqus_Mu/MSE/input_10000_samples.csv"  # 准备的10000个样本
    OUTPUT_DIR = "F:/Abaqus_Mu/MSE/predictions_10000"  # 输出目录

    # 特征和目标名称 (必须与训练时一致)
    FEATURE_NAMES = [
        'PGA', 'PGV', 'Sa(T1)', 'Es', 'nu', 'rho',
        'xi', 'c', 'phi', 'Ec', 'fy', 'tseg', 'mu'
    ]
    TARGET_NAMES = ['moment_ratio', 'diameter_strain_rate']

    # 需要log转换的特征 (必须与训练时一致)
    LOG_FEATURES = ['PGA', 'PGV', 'Sa(T1)', 'Es', 'c', 'phi', 'Ec', 'fy', 'tseg']

    # 数值稳定性
    EPS = 1e-8


# =============================================================================
# 2. 模型架构定义 (必须与训练时完全一致)
# =============================================================================

class MESModel(torch.nn.Module):
    """与训练代码完全一致的模型架构"""

    def __init__(
        self,
        input_dim: int,
        n_rbf: int,
        rbf_lengthscale: float,
        n_outputs: int,
        hidden_dims: list,
        dropout_rate: float = 0.1
    ):
        super(MESModel, self).__init__()

        self.input_dim = input_dim
        self.n_rbf = n_rbf
        self.rbf_lengthscale = rbf_lengthscale
        self.n_outputs = n_outputs

        # 总特征维度: 原始 + RBF
        feature_dim = input_dim + n_rbf

        # 共享编码器
        layers = []
        prev_dim = feature_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.encoder = torch.nn.Sequential(*layers)

        # 输出头
        self.mu_head = torch.nn.Linear(prev_dim, n_outputs)
        self.logvar_head = torch.nn.Linear(prev_dim, n_outputs)

    def forward(self, x_scaled: torch.Tensor, centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 计算RBF特征
        rbf_features = self._compute_rbf_features(x_scaled, centers)

        # 拼接原始和RBF特征
        features = torch.cat([x_scaled, rbf_features], dim=1)

        # 编码
        encoded = self.encoder(features)

        # 预测均值和对数方差
        mu = self.mu_head(encoded)
        logvar = self.logvar_head(encoded)

        return mu, logvar

    def _compute_rbf_features(self, x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """计算RBF核特征"""
        diff = x.unsqueeze(1) - centers.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        rbf = torch.exp(-0.5 * dist_sq / (self.rbf_lengthscale ** 2))
        return rbf


# =============================================================================
# 3. 数据预处理
# =============================================================================

def preprocess_input_data(
    df: pd.DataFrame,
    x_scaler,
    feature_names: list,
    log_features: list
) -> np.ndarray:
    """
    预处理输入数据 (与训练时保持一致的转换)

    Args:
        df: 包含特征的DataFrame
        x_scaler: 训练时保存的StandardScaler
        feature_names: 特征名称列表
        log_features: 需要log转换的特征

    Returns:
        标准化后的特征矩阵
    """
    print("\n" + "=" * 60)
    print("预处理输入数据...")
    print("=" * 60)

    # 检查列是否完整
    missing_cols = set(feature_names) - set(df.columns)
    if missing_cols:
        raise ValueError(f"输入CSV缺少列: {missing_cols}")

    # 提取特征
    X = df[feature_names].copy()

    # Log转换
    print(f"\n对以下特征进行log转换: {log_features}")
    for col in log_features:
        if col in X.columns:
            # 检查非正值
            if (X[col] <= 0).any():
                print(f"  警告: {col} 存在非正值,添加偏移量")
                X[col] = X[col] - X[col].min() + 1e-6

            X[col] = np.log(X[col])

    # 标准化
    X_scaled = x_scaler.transform(X.values)

    print(f"\n✅ 预处理完成! 数据形状: {X_scaled.shape}")

    return X_scaled


# =============================================================================
# 4. 批量预测
# =============================================================================

@torch.no_grad()
def batch_predict(
    model: MESModel,
    X_scaled: np.ndarray,
    centers: torch.Tensor,
    device: torch.device,
    batch_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量预测 (避免内存溢出)

    Args:
        model: 训练好的模型
        X_scaled: 标准化后的输入特征
        centers: RBF中心
        device: 计算设备
        batch_size: 批次大小

    Returns:
        pred_mu_scaled: 预测均值 (标准化空间)
        pred_logvar_scaled: 预测log方差 (标准化空间)
    """
    print("\n" + "=" * 60)
    print("开始批量预测...")
    print("=" * 60)

    model.eval()

    n_samples = len(X_scaled)
    n_batches = int(np.ceil(n_samples / batch_size))

    pred_mu_list = []
    pred_logvar_list = []

    for i in range(n_batches):
        # 获取批次数据
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)

        X_batch = torch.FloatTensor(X_scaled[start_idx:end_idx]).to(device)

        # 预测
        mu, logvar = model(X_batch, centers)

        pred_mu_list.append(mu.cpu().numpy())
        pred_logvar_list.append(logvar.cpu().numpy())

        # 打印进度
        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            print(f"  进度: {i + 1}/{n_batches} 批次 ({end_idx}/{n_samples} 样本)")

    pred_mu = np.vstack(pred_mu_list)
    pred_logvar = np.vstack(pred_logvar_list)

    print(f"\n✅ 预测完成! 输出形状: {pred_mu.shape}")

    return pred_mu, pred_logvar


# =============================================================================
# 5. 结果转换到原始空间
# =============================================================================

def transform_predictions_to_original_space(
    pred_mu_scaled: np.ndarray,
    pred_logvar_scaled: np.ndarray,
    y_scaler,
    eps: float = 1e-8
) -> Dict[str, np.ndarray]:
    """
    将标准化空间的预测转换回原始空间

    模型输出在 y' = ln(y) 的标准化空间
    需要:
        1. 反标准化到 y' (log空间)
        2. 应用对数正态变换到原始空间

    对于 Y ~ LogNormal(μ', σ'²):
        E[Y] = exp(μ' + σ'²/2)
        Var[Y] = exp(2μ' + σ'²) * (exp(σ'²) - 1)
        95% CI = exp(μ' ± 1.96*σ')

    Returns:
        包含以下键的字典:
        - E_Y: 原始空间的期望值
        - Std_Y: 原始空间的标准差
        - CI_lower: 95%置信区间下界
        - CI_upper: 95%置信区间上界
        - mu_logspace: log空间的均值
        - sigma_logspace: log空间的标准差
    """
    print("\n" + "=" * 60)
    print("转换预测结果到原始空间...")
    print("=" * 60)

    # 1. 反标准化到log空间
    mu_logspace = y_scaler.inverse_transform(pred_mu_scaled)

    # 2. 调整log方差
    # 因为 Var(y_scaled) = Var(y) / std²
    # 所以 logvar_y = logvar_scaled + 2*log(std)
    std_y = y_scaler.scale_  # shape: (n_outputs,)
    logvar_logspace = pred_logvar_scaled + 2.0 * np.log(std_y[None, :])

    sigma2_logspace = np.exp(logvar_logspace)
    sigma_logspace = np.sqrt(sigma2_logspace)

    # 3. 对数正态变换到原始空间
    E_Y = np.exp(mu_logspace + 0.5 * sigma2_logspace)
    Var_Y = np.exp(2 * mu_logspace + sigma2_logspace) * (np.exp(sigma2_logspace) - 1.0)
    Std_Y = np.sqrt(Var_Y)

    # 4. 95%置信区间
    CI_lower = np.exp(mu_logspace - 1.96 * sigma_logspace)
    CI_upper = np.exp(mu_logspace + 1.96 * sigma_logspace)

    print(f"✅ 转换完成!")
    print(f"  期望值范围: [{E_Y.min():.4f}, {E_Y.max():.4f}]")
    print(f"  标准差范围: [{Std_Y.min():.4f}, {Std_Y.max():.4f}]")

    return {
        'E_Y': E_Y,                      # 期望值 (原始空间)
        'Std_Y': Std_Y,                  # 标准差 (原始空间)
        'CI_lower': CI_lower,            # 95% CI下界
        'CI_upper': CI_upper,            # 95% CI上界
        'mu_logspace': mu_logspace,      # log空间均值
        'sigma_logspace': sigma_logspace # log空间标准差
    }


# =============================================================================
# 6. 保存结果
# =============================================================================

def save_predictions(
        input_df: pd.DataFrame,
        predictions: Dict[str, np.ndarray],
        target_names: list,
        output_dir: str
):
    """
    保存预测结果到CSV (增加了保存Log空间参数的功能)
    """
    print("\n" + "=" * 60)
    print("保存预测结果...")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 创建结果DataFrame
    result_df = input_df.copy()

    # 添加预测结果
    for i, name in enumerate(target_names):
        # 1. 原始空间结果 (物理意义直观)
        result_df[f'{name}_pred_mean'] = predictions['E_Y'][:, i]
        result_df[f'{name}_pred_std'] = predictions['Std_Y'][:, i]
        result_df[f'{name}_pred_CI_lower'] = predictions['CI_lower'][:, i]
        result_df[f'{name}_pred_CI_upper'] = predictions['CI_upper'][:, i]

        # 2. Log空间结果 (用于Copula易损性计算，这部分是修正的关键！)
        # 确保这些键在 transform_predictions_to_original_space 的返回值里存在
        result_df[f'{name}_mu_log'] = predictions['mu_logspace'][:, i]
        result_df[f'{name}_sigma_log'] = predictions['sigma_logspace'][:, i]

        # 计算变异系数 (CV%)
        cv = predictions['Std_Y'][:, i] / (predictions['E_Y'][:, i] + 1e-8) * 100
        result_df[f'{name}_CV_percent'] = cv

    # 保存完整结果
    # 你的代码之前可能覆盖了，这里确保文件名正确
    full_output_path = os.path.join(output_dir, 'predictions_10000.csv')
    result_df.to_csv(full_output_path, index=False)
    print(f"✅ 完整结果已保存: {full_output_path}")

    # (可选) 打印前几行看看有没有这两列
    print("检查生成的列名:", [col for col in result_df.columns if 'log' in col])

    # 保存统计摘要
    summary_data = []
    for i, name in enumerate(target_names):
        summary_data.append({
            'Variable': name,
            'Mean_Prediction': predictions['E_Y'][:, i].mean(),
            'Std_Prediction': predictions['E_Y'][:, i].std(),
            'Min_Prediction': predictions['E_Y'][:, i].min(),
            'Max_Prediction': predictions['E_Y'][:, i].max(),
            'Mean_Uncertainty_Std': predictions['Std_Y'][:, i].mean(),
            'Mean_CV_percent': cv.mean()
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'prediction_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ 统计摘要已保存: {summary_path}")

    print("\n" + "=" * 60)
    print("预测统计摘要:")
    print("=" * 60)
    print(summary_df.to_string(index=False))


# =============================================================================
# 7. 主函数
# =============================================================================

def main():
    """主执行流程"""
    cfg = PredictionConfig()

    print("\n" + "=" * 80)
    print("MES模型批量预测工具 - 10000样本")
    print("=" * 80)

    # 1. 检查文件
    print("\n【步骤 1/6】检查文件...")
    if not os.path.exists(cfg.MODEL_PATH):
        raise FileNotFoundError(f"模型文件未找到: {cfg.MODEL_PATH}")
    if not os.path.exists(cfg.INPUT_CSV):
        raise FileNotFoundError(f"输入CSV未找到: {cfg.INPUT_CSV}")
    print("✅ 文件检查通过")

    # 2. 加载模型
    print("\n【步骤 2/6】加载训练好的模型...")
    checkpoint = torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE)

    # 提取模型配置
    x_scaler = checkpoint['x_scaler']
    y_scaler = checkpoint['y_scaler']
    centers_np = checkpoint['centers']
    hyperparams = checkpoint.get('hyperparameters', {})

    # 重建模型架构
    model = MESModel(
        input_dim=len(cfg.FEATURE_NAMES),
        n_rbf=hyperparams.get('n_rbf', centers_np.shape[0]),
        rbf_lengthscale=hyperparams.get('rbf_lengthscale', 1.0),
        n_outputs=len(cfg.TARGET_NAMES),
        hidden_dims=[
            hyperparams.get('hidden_dim1', 128),
            hyperparams.get('hidden_dim2', 64)
        ],
        dropout_rate=hyperparams.get('dropout_rate', 0.1)
    ).to(cfg.DEVICE)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    centers_torch = torch.FloatTensor(centers_np).to(cfg.DEVICE)

    print(f"✅ 模型加载成功")
    print(f"  - RBF中心数: {centers_np.shape[0]}")
    print(f"  - 隐藏层维度: {[hyperparams.get('hidden_dim1', 128), hyperparams.get('hidden_dim2', 64)]}")

    # 3. 读取输入数据
    print("\n【步骤 3/6】读取输入数据...")
    input_df = pd.read_csv(cfg.INPUT_CSV)
    print(f"✅ 读取成功: {len(input_df)} 个样本")
    print(f"  数据形状: {input_df.shape}")
    print(f"  列名: {list(input_df.columns)}")

    # 4. 预处理
    X_scaled = preprocess_input_data(
        input_df, x_scaler, cfg.FEATURE_NAMES, cfg.LOG_FEATURES
    )

    # 5. 批量预测
    print("\n【步骤 5/6】批量预测...")
    pred_mu_scaled, pred_logvar_scaled = batch_predict(
        model, X_scaled, centers_torch, cfg.DEVICE, batch_size=256
    )

    # 6. 转换到原始空间
    predictions = transform_predictions_to_original_space(
        pred_mu_scaled, pred_logvar_scaled, y_scaler, cfg.EPS
    )

    # 7. 保存结果
    print("\n【步骤 6/6】保存结果...")
    save_predictions(
        input_df, predictions, cfg.TARGET_NAMES, cfg.OUTPUT_DIR
    )

    # 完成
    print("\n" + "=" * 80)
    print("🎉 预测完成!")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  1. 完整结果 (输入+预测): {cfg.OUTPUT_DIR}/predictions_full.csv")
    print(f"  2. 仅预测结果: {cfg.OUTPUT_DIR}/predictions_only.csv")
    print(f"  3. 统计摘要: {cfg.OUTPUT_DIR}/prediction_summary.csv")
    print(f"\n每个目标变量包含:")
    print(f"  - {cfg.TARGET_NAMES[0]}_pred_mean: 预测均值")
    print(f"  - {cfg.TARGET_NAMES[0]}_pred_std: 预测标准差")
    print(f"  - {cfg.TARGET_NAMES[0]}_pred_CI_lower: 95%置信区间下界")
    print(f"  - {cfg.TARGET_NAMES[0]}_pred_CI_upper: 95%置信区间上界")
    print(f"  - {cfg.TARGET_NAMES[0]}_CV_percent: 变异系数(%)")


# =============================================================================
# 8. 入口
# =============================================================================

if __name__ == "__main__":
    main()
