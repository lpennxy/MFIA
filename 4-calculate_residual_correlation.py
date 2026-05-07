# calculate_residual_correlation.py
"""
计算训练集上的残差相关系数，用于后续Copula易损性分析
在物理对数空间计算标准化残差。
"""

import os
import numpy as np
import pandas as pd
import torch
import joblib
import scipy.stats as stats  # 引入 scipy 用于计算 Kendall's tau


# 引入你的 MESModel 类定义 (保持不变)
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


# 配置 (保持不变)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "F:/Abaqus_Mu/MSE/results_2/best_mes_model.pt"
TRAIN_DATA_PATH = "F:/Abaqus_Mu/MSE/MSE_TASK_2.csv"
FEATURE_NAMES = ['PGA', 'PGV', 'Sa(T1)', 'Es', 'nu', 'rho', 'xi', 'c', 'phi', 'Ec', 'fy', 'tseg', 'mu']
TARGET_NAMES = ['moment_ratio', 'diameter_strain_rate']
LOG_FEATURES = ['PGA', 'PGV', 'Sa(T1)', 'Es', 'c', 'phi', 'Ec', 'fy', 'tseg']


def main():
    print("计算残差相关系数 (修正版)...")

    # 1. 加载模型 (保持不变)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    x_scaler = checkpoint['x_scaler']
    y_scaler = checkpoint['y_scaler']  # 这个 scaler 非常重要
    centers = torch.FloatTensor(checkpoint['centers']).to(DEVICE)

    model = MESModel(
        input_dim=len(FEATURE_NAMES),
        n_rbf=checkpoint['hyperparameters'].get('n_rbf', centers.shape[0]),
        rbf_lengthscale=checkpoint['hyperparameters'].get('rbf_lengthscale', 1.0),
        n_outputs=len(TARGET_NAMES),
        hidden_dims=[checkpoint['hyperparameters'].get('hidden_dim1', 128),
                     checkpoint['hyperparameters'].get('hidden_dim2', 64)],
        dropout_rate=0.0
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. 读取训练数据 (保持不变)
    df = pd.read_csv(TRAIN_DATA_PATH)

    # 3. 预处理输入 (保持不变)
    X = df[FEATURE_NAMES].copy()
    for col in LOG_FEATURES:
        X[col] = np.log(X[col])
    X_scaled = x_scaler.transform(X.values)
    X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)

    # 4. 预处理真实标签 (转换到 Log 空间，但不标准化!)
    Y_true = df[TARGET_NAMES].values
    for i in range(Y_true.shape[1]):
        if (Y_true[:, i] <= 0).any():
            Y_true[:, i] = Y_true[:, i] - Y_true[:, i].min() + 1e-6
    Y_true_log = np.log(Y_true)

    # --- [关键修改点开始] ---
    # 删除旧代码： Y_true_scaled = y_scaler.transform(Y_true_log)

    # 5. 预测并反变换到物理对数空间
    with torch.no_grad():
        mu_pred_scaled_tensor, logvar_pred_scaled_tensor = model(X_tensor, centers)

        mu_pred_scaled = mu_pred_scaled_tensor.cpu().numpy()
        logvar_pred_scaled = logvar_pred_scaled_tensor.cpu().numpy()

    # A. 反变换均值 mu_log = mu_scaled * scale + mean
    mu_log_pred = y_scaler.inverse_transform(mu_pred_scaled)

    # B. 反变换方差 sigma_log^2 = sigma_scaled^2 * scale^2
    # log(sigma_log^2) = log(sigma_scaled^2) + 2*log(scale)
    # log(sigma_scaled^2) 就是网络的输出 logvar_pred_scaled
    std_y_base = y_scaler.scale_  # 原始数据的标准差
    logvar_log_pred = logvar_pred_scaled + 2.0 * np.log(std_y_base[None, :])
    sigma_log_pred = np.sqrt(np.exp(logvar_log_pred))

    # 6. 在物理对数空间计算标准化残差
    # z = (y_true_log - mu_log_pred) / sigma_log_pred
    residuals = (Y_true_log - mu_log_pred) / sigma_log_pred

    print(f"残差计算完成。Shape: {residuals.shape}")
    # 检查一下残差的性质，应该接近均值0，方差1
    print(f"残差均值 (期望~0): {residuals.mean(axis=0)}")
    print(f"残差标准差 (期望~1): {residuals.std(axis=0)}")
    # --- [关键修改点结束] ---

    # 7. 计算相关矩阵 (使用 Pearson 或 Kendall)
    # 对于 Copula，通常推荐使用基于秩的 Kendall's tau，然后再映射到 Gaussian Copula 的 rho
    # 这样对异常值和非线性更稳健。

    # 方法 A: 直接计算 Pearson 线性相关 (你原来的方法，受极端值影响大)
    # corr_matrix_pearson = np.corrcoef(residuals, rowvar=False)
    # rho_pearson = corr_matrix_pearson[0, 1]
    # print(f"Pearson 相关系数: {rho_pearson:.4f}")

    # 方法 B: 计算 Kendall's tau 并映射 (更推荐，与论文理论一致)
    tau, p_value = stats.kendalltau(residuals[:, 0], residuals[:, 1])
    # 对于高斯 Copula，rho = sin(pi/2 * tau)
    rho_kendall_mapped = np.sin(np.pi / 2 * tau)

    print("-" * 40)
    print(f"目标变量 1: {TARGET_NAMES[0]} (Moment Ratio)")
    print(f"目标变量 2: {TARGET_NAMES[1]} (Diameter Strain Rate)")
    print(f"Kendall's Tau: {tau:.4f}")
    print(f"映射后的高斯 Copula 相关系数 (rho): {rho_kendall_mapped:.4f}")
    print("-" * 40)

    # 使用映射后的 rho 保存
    final_rho = rho_kendall_mapped

    with open("residual_correlation.txt", "w") as f:
        f.write(str(final_rho))
    print(f"最终结果 (rho={final_rho:.4f}) 已保存到 residual_correlation.txt")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.scatter(residuals[:, 0], residuals[:, 1], alpha=0.5, s=10)
    plt.xlabel(f"Residuals of {TARGET_NAMES[0]}")
    plt.ylabel(f"Residuals of {TARGET_NAMES[1]}")
    plt.title(f"Residual Scatter Plot (Kendall tau = {tau:.4f})")
    plt.grid(True)
    # 画一条对角线参考
    lims = [np.min(residuals), np.max(residuals)]
    plt.plot(lims, lims, 'r--', alpha=0.3, label='Perfect Correlation')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
