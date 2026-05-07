# 5-probabilistic_modeling.py
"""
实现论文 2.3 节：边缘分布非参数拟合与依赖结构建模
1. 计算训练集的标准化残差，构建 ABKDE 模型 (Moment Anchoring)。
2. 构建基于残差相关性的 Gaussian Copula 模型。
3. 保存这些概率模型对象，供 2.4 节易损性分析使用。
4. 保存绘图源数据，并生成出版级质量插图。
"""

import os
import numpy as np
import pandas as pd
import torch
import joblib
import scipy.stats as stats
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


# =============================================================================
# 0. 配置与模型定义
# =============================================================================

class ProbConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输入路径 (请确保这些路径存在或根据需要修改)
    MODEL_PATH = "F:/Abaqus_Mu/MSE/results_2/best_mes_model.pt"
    TRAIN_DATA_PATH = "F:/Abaqus_Mu/MSE/MSE_TASK_2.csv"
    CORRELATION_FILE = "residual_correlation.txt"
    PRED_10000_PATH = "F:/Abaqus_Mu/MSE/predictions_10000/predictions_10000.csv"

    # 输出路径
    OUTPUT_DIR = "F:/Abaqus_Mu/MSE/probabilistic_models"
    FIGURE_DIR = "F:/Abaqus_Mu/MSE/probabilistic_models/figures_chapter4"
    PLOT_DATA_DIR = "F:/Abaqus_Mu/MSE/probabilistic_models/plot_data"  
    MODEL_SAVE_NAME = "F:/Abaqus_Mu/MSE/probabilistic_models/probabilistic_model_suite.pkl"

    # 特征与目标
    FEATURE_NAMES = ['PGA', 'PGV', 'Sa(T1)', 'Es', 'nu', 'rho', 'xi', 'c', 'phi', 'Ec', 'fy', 'tseg', 'mu']
    TARGET_NAMES = ['moment_ratio', 'diameter_strain_rate']

    # 论文用标签 (使用 LaTeX 格式)
    TARGET_LABELS = {
        'moment_ratio': r'Moment Ratio $M/M_y$',
        'diameter_strain_rate': r'Diameter Strain Rate $\theta_{max}$'
    }
    LOG_FEATURES = ['PGA', 'PGV', 'Sa(T1)', 'Es', 'c', 'phi', 'Ec', 'fy', 'tseg']


# 保持原模型定义不变
class MESModel(torch.nn.Module):
    def __init__(self, input_dim, n_rbf, rbf_lengthscale, n_outputs, hidden_dims, dropout_rate=0.1):
        super(MESModel, self).__init__()
        self.rbf_lengthscale = rbf_lengthscale
        feature_dim = input_dim + n_rbf
        layers = []
        prev_dim = feature_dim
        for hidden_dim in hidden_dims:
            layers.extend([torch.nn.Linear(prev_dim, hidden_dim), torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU(),
                           torch.nn.Dropout(dropout_rate)])
            prev_dim = hidden_dim
        self.encoder = torch.nn.Sequential(*layers)
        self.mu_head = torch.nn.Linear(prev_dim, n_outputs)
        self.logvar_head = torch.nn.Linear(prev_dim, n_outputs)

    def forward(self, x_scaled, centers):
        diff = x_scaled.unsqueeze(1) - centers.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        rbf_features = torch.exp(-0.5 * dist_sq / (self.rbf_lengthscale ** 2))
        features = torch.cat([x_scaled, rbf_features], dim=1)
        encoded = self.encoder(features)
        return self.mu_head(encoded), self.logvar_head(encoded)



# =============================================================================
# 1. 科学绘图风格设置 (修正版：兼容性增强)
# =============================================================================

def set_publication_style():
    """配置 Matplotlib 以生成符合学术期刊标准的图片"""

    # --- 1. 智能选择基础样式 (解决 OSError 问题) ---
    # 获取当前环境支持的所有样式
    available_styles = plt.style.available

    # 优先尝试不同版本的 seaborn paper 样式
    if 'seaborn-v0_8-paper' in available_styles:
        plt.style.use('seaborn-v0_8-paper')
    elif 'seaborn-paper' in available_styles:
        plt.style.use('seaborn-paper')
    else:
        # 如果以上都没有，使用经典样式作为基础
        plt.style.use('fast')

        # --- 2. 强制覆盖为出版级参数 ---
    # 这里的设置优先级最高，会覆盖上面的基础样式
    params = {
        'font.family': 'serif',
        # 尝试通过列表指定字体，Windows下通常有 Times New Roman
        'font.serif': ['Times New Roman', 'SimSun', 'DejaVu Serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.dpi': 300,

        # 线条与刻度
        'axes.linewidth': 1.0,  # 坐标轴线宽
        'grid.linewidth': 0.5,  # 网格线宽
        'lines.linewidth': 1.5,  # 绘图线宽
        'xtick.direction': 'in',  # 刻度朝内
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,

        # 布局优化
        'figure.constrained_layout.use': True,
        'mathtext.fontset': 'stix',  # 让公式字体更像 LaTeX
        'text.usetex': False  # 保持 False 以避免复杂的 LaTeX 环境配置报错
    }
    mpl.rcParams.update(params)



# =============================================================================
# 2. 核心类定义 (ABKDE & Copula)
# =============================================================================

class MomentAnchoredKDE:
    def __init__(self, target_name):
        self.target_name = target_name
        self.kde = None
        self.residuals_std = None

    def fit(self, residuals):
        self.kde = gaussian_kde(residuals)
        self.residuals_std = residuals
        print(f"[{self.target_name}] KDE 拟合完成. 因子: {self.kde.factor:.4f}")

    def get_cdf(self, y_val, mu_cond, sigma_cond):
        z_val = (y_val - mu_cond) / sigma_cond
        if isinstance(z_val, (int, float)):
            z_val = np.array([z_val])
        # 使用 integrate_box_1d 较慢，生产环境建议使用预计算插值
        u_values = np.array([self.kde.integrate_box_1d(-np.inf, z) for z in z_val])
        return np.clip(u_values, 1e-10, 1.0 - 1e-10)


class GaussianCopulaModel:
    def __init__(self, rho):
        self.rho = rho
        self.R = np.array([[1.0, rho], [rho, 1.0]])
        self.mv_norm = stats.multivariate_normal(mean=[0, 0], cov=self.R)

    def joint_probability_series(self, u1, u2):
        z1 = stats.norm.ppf(u1)
        z2 = stats.norm.ppf(u2)
        points = np.column_stack((z1, z2))
        joint_cdf = self.mv_norm.cdf(points)
        return 1.0 - joint_cdf

    def joint_probability_parallel(self, u1, u2):
        p1 = 1.0 - u1
        p2 = 1.0 - u2
        z1 = stats.norm.ppf(p1)
        z2 = stats.norm.ppf(p2)
        points = np.column_stack((z1, z2))
        return self.mv_norm.cdf(points)


# =============================================================================
# 3. 辅助函数：提取残差 & 保存绘图数据
# =============================================================================

def get_training_residuals(cfg):
    """计算标准化残差"""
    print("\n[Step 1] 计算训练集残差...")

    # 加载模型
    checkpoint = torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE)
    x_scaler = checkpoint['x_scaler']
    y_scaler = checkpoint['y_scaler']
    centers = torch.FloatTensor(checkpoint['centers']).to(cfg.DEVICE)
    hyperparams = checkpoint.get('hyperparameters', {})

    model = MESModel(
        input_dim=len(cfg.FEATURE_NAMES),
        n_rbf=hyperparams.get('n_rbf', centers.shape[0]),
        rbf_lengthscale=hyperparams.get('rbf_lengthscale', 1.0),
        n_outputs=len(cfg.TARGET_NAMES),
        hidden_dims=[hyperparams.get('hidden_dim1', 128), hyperparams.get('hidden_dim2', 64)]
    ).to(cfg.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 处理数据
    df = pd.read_csv(cfg.TRAIN_DATA_PATH)
    X = df[cfg.FEATURE_NAMES].copy()
    for col in cfg.LOG_FEATURES: X[col] = np.log(X[col])
    X_scaled = x_scaler.transform(X.values)
    X_tensor = torch.FloatTensor(X_scaled).to(cfg.DEVICE)

    Y_true = df[cfg.TARGET_NAMES].values
    for i in range(Y_true.shape[1]):
        if (Y_true[:, i] <= 0).any(): Y_true[:, i] = Y_true[:, i] - Y_true[:, i].min() + 1e-6
    Y_true_log = np.log(Y_true)

    with torch.no_grad():
        mu_scaled, logvar_scaled = model(X_tensor, centers)

    mu_scaled_np = mu_scaled.cpu().numpy()
    logvar_scaled_np = logvar_scaled.cpu().numpy()

    # 逆变换
    mu_log = y_scaler.inverse_transform(mu_scaled_np)
    std_y_scale = y_scaler.scale_
    logvar_log = logvar_scaled_np + 2.0 * np.log(std_y_scale[None, :])
    sigma_log = np.sqrt(np.exp(logvar_log))

    residuals = (Y_true_log - mu_log) / sigma_log
    return residuals


def save_plotting_data(residuals, target_names, save_dir):
    """
    [关键步骤] 保存用于绘图的源数据。
    这样以后即使不运行模型，也可以直接加载 CSV 来调整绘图代码。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. 保存残差数据
    df_res = pd.DataFrame(residuals, columns=target_names)
    csv_path = os.path.join(save_dir, "plot_data_residuals.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\n[Data Saving] 残差绘图数据已保存至: {csv_path}")

    # 2. 保存相关性矩阵数据
    corr_matrix = df_res.corr(method='kendall')  # Kendall tau
    # 转换为 Pearson rho (Gaussian Copula 假设)
    rho_matrix = np.sin(np.pi / 2 * corr_matrix)
    rho_path = os.path.join(save_dir, "plot_data_correlation_matrix.csv")
    rho_matrix.to_csv(rho_path)
    print(f"[Data Saving] 相关性矩阵数据已保存至: {rho_path}")

    return df_res, rho_matrix


# =============================================================================
# 4. 出版级绘图函数
# =============================================================================

def plot_residual_diagnostics(residuals_df, target_labels, save_dir):
    """
    绘制图 7：残差分布诊断 (直方图 + KDE + Q-Q 图)
    """
    set_publication_style()  # 应用样式

    n_targets = residuals_df.shape[1]
    target_names = residuals_df.columns

    # 创建画布：宽 10 英寸，高 8 英寸 (适合 A4 论文双栏跨栏)
    fig = plt.figure(figsize=(10, 4 * n_targets))
    # 使用 GridSpec 进行精细布局控制
    gs = fig.add_gridspec(n_targets, 2, width_ratios=[1, 1], hspace=0.3, wspace=0.2)

    z_grid = np.linspace(-4, 4, 500)
    pdf_norm = stats.norm.pdf(z_grid)

    # 统一颜色方案
    hist_color = '#5DADE2'  # 柔和蓝
    kde_color = '#E74C3C'  # 砖红色
    ref_color = 'k'  # 黑色

    for i, name in enumerate(target_names):
        res = residuals_df[name].values
        label_text = target_labels.get(name, name)

        # --- 左图：分布直方图 ---
        ax_hist = fig.add_subplot(gs[i, 0])

        # 绘制直方图 (使用 matplotlib 原生 hist 更易控制样式)
        ax_hist.hist(res, bins=30, density=True, color=hist_color, alpha=0.5,
                     edgecolor='white', linewidth=0.5, label='Residuals')

        # 绘制 KDE
        kde = gaussian_kde(res)
        ax_hist.plot(z_grid, kde(z_grid), color=kde_color, linewidth=2, label='ABKDE Fit')

        # 绘制标准正态参考线
        ax_hist.plot(z_grid, pdf_norm, color=ref_color, linestyle='--', linewidth=1.5,
                     label=r'Standard Normal $\mathcal{N}(0,1)$')

        ax_hist.set_xlabel('Standardized Residual $z$')
        ax_hist.set_ylabel('Probability Density')
        ax_hist.set_xlim([-4.5, 4.5])
        ax_hist.legend(frameon=True, fancybox=False, edgecolor='k', framealpha=1.0, fontsize=10)
        ax_hist.text(0.02, 0.95, f'({chr(97 + i * 2)}) {label_text}', transform=ax_hist.transAxes,
                     fontweight='bold', va='top')  # 子图编号 (a), (c)...

        # --- 右图：Q-Q 图 ---
        ax_qq = fig.add_subplot(gs[i, 1])
        (osm, osr), (slope, intercept, r) = stats.probplot(res, dist="norm", plot=None)

        ax_qq.scatter(osm, osr, s=20, alpha=0.6, facecolors='none', edgecolors='#2E86C1', label='Sample Quantiles')
        ax_qq.plot(osm, slope * osm + intercept, color='#E74C3C', linewidth=2, label='Theoretical Reference')

        ax_qq.set_xlabel('Theoretical Quantiles')
        ax_qq.set_ylabel('Ordered Residuals')
        ax_qq.text(0.02, 0.95, f'({chr(97 + i * 2 + 1)}) Q-Q Plot', transform=ax_qq.transAxes,
                   fontweight='bold', va='top')  # 子图编号 (b), (d)...

        # Jarque-Bera 检验标注
        jb_stat, jb_p = stats.jarque_bera(res)
        stats_text = f'Jarque-Bera Test:\n$p$-value = {jb_p:.2e}'
        ax_qq.text(0.65, 0.1, stats_text, transform=ax_qq.transAxes, fontsize=10,
                   bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    save_path = os.path.join(save_dir, "Figure7_Residual_Diagnostics.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "Figure7_Residual_Diagnostics.pdf"), bbox_inches='tight')  # 保存PDF矢量图
    print(f"[Plotting] 图7 已保存至: {save_path}")
    plt.close()


def plot_residual_copula_heatmap(rho_df, target_labels, save_dir):
    """
    绘制图 8：残差相关性热力图
    """
    set_publication_style()

    # 替换索引和列名为 LaTeX 标签
    labels = [target_labels.get(name, name) for name in rho_df.columns]

    plt.figure(figsize=(6, 5))

    # 绘制热力图
    # 使用 diverging colormap (vlag 或 coolwarm) 来强调正负相关
    mask = np.triu(np.ones_like(rho_df, dtype=bool), k=1)  # 如果想要只显示下三角，可以去掉这行的注释并传给 heatmap mask参数

    heatmap = sns.heatmap(rho_df, vmin=-1, vmax=1, annot=True, fmt=".2f",
                          cmap='coolwarm', square=True,
                          linewidths=1.0, linecolor='black',
                          cbar_kws={"shrink": .8, "label": r"Correlation Coefficient $\rho$"},
                          xticklabels=labels, yticklabels=labels)

    # 调整坐标轴字体
    plt.xticks(rotation=0)
    plt.yticks(rotation=90, va='center')
    plt.title('Residual Correlation Structure\n(Implied Gaussian Copula)', pad=15, fontweight='bold')

    save_path = os.path.join(save_dir, "Figure8_Copula_Heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "Figure8_Copula_Heatmap.pdf"), bbox_inches='tight')
    print(f"[Plotting] 图8 已保存至: {save_path}")
    plt.close()


# =============================================================================
# 5. 主流程
# =============================================================================

def main():
    cfg = ProbConfig()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.FIGURE_DIR, exist_ok=True)
    os.makedirs(cfg.PLOT_DATA_DIR, exist_ok=True)  # 确保数据目录存在

    # 1. 获取训练集残差
    residuals_raw = get_training_residuals(cfg)

    # 2. 保存绘图数据 (这是新增的关键步骤)
    #    现在我们有了 df_res 和 rho_matrix，接下来的绘图都基于这些 DataFrame
    df_res, df_rho = save_plotting_data(residuals_raw, cfg.TARGET_NAMES, cfg.PLOT_DATA_DIR)

    # 3. 绘制出版级插图
    print("\n[Step 2] 绘制论文插图 (基于已保存的数据)...")
    plot_residual_diagnostics(df_res, cfg.TARGET_LABELS, cfg.FIGURE_DIR)
    plot_residual_copula_heatmap(df_rho, cfg.TARGET_LABELS, cfg.FIGURE_DIR)

    # 4. 构建并保存模型对象
    print("\n[Step 3] 构建并打包概率模型...")
    abkde_models = {}
    for name in cfg.TARGET_NAMES:
        kde_model = MomentAnchoredKDE(target_name=name)
        # 注意：这里直接用保存好的 df 列数据
        kde_model.fit(df_res[name].values)
        abkde_models[name] = kde_model

    # 获取非对角线相关系数 (假设2个变量)
    if df_rho.shape[0] >= 2:
        rho_final = df_rho.iloc[0, 1]  # 取 (0,1) 元素
        print(f"  Copula 参数 rho: {rho_final:.4f}")
    else:
        rho_final = 0.0

    copula_model = GaussianCopulaModel(rho_final)

    model_suite = {
        'abkde_moment': abkde_models['moment_ratio'],
        'abkde_strain': abkde_models['diameter_strain_rate'],
        'copula': copula_model,
        'feature_names': cfg.FEATURE_NAMES,
        'target_names': cfg.TARGET_NAMES,
        'target_labels': cfg.TARGET_LABELS
    }

    joblib.dump(model_suite, cfg.MODEL_SAVE_NAME)
    print(f"✅ 完成！模型已保存至: {cfg.MODEL_SAVE_NAME}")
    print(f"📊 绘图源数据已保存至: {cfg.PLOT_DATA_DIR} (可随时取用)")


if __name__ == "__main__":
    main()
