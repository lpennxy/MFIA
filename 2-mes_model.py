# mes_training_complete.py
"""
Multi-output Heteroscedastic Neural Network with Physics-Informed Constraints
for Modal Engineering Seismic (MES) Response Prediction

Features:
- Log-normal transformation for input/output variables
- RBF kernel features for nonlinearity
- Physics-informed monotonicity constraints
- Heteroscedastic uncertainty quantification
- Multi-output prediction (moment ratio & diameter strain rate)
"""

import os
import random
import math
import warnings
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
import optuna

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')


# =============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# =============================================================================

class Config:
    """Central configuration class"""
    # Random seed
    SEED = 42

    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data configuration
    # TODO: 根据你的实际CSV列名修改
    FEATURE_NAMES = [
        'PGA', 'PGV', 'Sa(T1)', 'Es', 'nu', 'rho',
        'xi', 'c', 'phi', 'Ec', 'fy', 'tseg', 'mu'
    ]
    TARGET_NAMES = ['moment_ratio', 'diameter_strain_rate']

    # Features to apply log transformation (LogNormal distributed)
    # TODO: 根据你的数据分布调整
    LOG_FEATURES = ['PGA', 'PGV', 'Sa(T1)', 'Es', 'c', 'phi', 'Ec', 'fy', 'tseg']

    # Physical monotonicity expectations
    # +1: increasing, -1: decreasing, None: no constraint
    # TODO: 根据物理知识验证和调整
    MONOTONICITY_EXPECTATION = {
        'PGA': +1,  # 地震强度增大 -> 响应增大
        'PGV': +1,  # 速度增大 -> 响应增大
        'Sa(T1)': +1,  # 谱加速度增大 -> 响应增大
        'Es': None,  # 土体模量增大 -> 响应减小-1
        'Ec': None,  # 混凝土模量增大 -> 响应减小
        'xi': None,  # 阻尼增大 -> 响应减小
        'c': None,  # 粘聚力增大 -> 响应减小-1
        'phi': None,  # 摩擦角增大 -> 响应减小-1
        'fy': None,  # 屈服强度增大 -> 响应减小-1
        'tseg': None,  # 截面厚度增大 -> 响应减小
        'mu': None,  # 延性增大 -> 响应减小
        'nu': None,  # 泊松比影响不确定
        'rho': None  # 密度影响不确定
    }

    # Model architecture
    N_RBF = 40  # Number of RBF centers
    RBF_LENGTHSCALE = 2.0  # RBF kernel lengthscale
    HIDDEN_DIMS = [128, 64]  # Hidden layer dimensions
    DROPOUT_RATE = 0.1  # Dropout for regularization

    # Training parameters
    N_EPOCHS = 300
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    LAMBDA_PHYS = 0.1  # Physics penalty weight (调参关键)
    EARLY_STOPPING_PATIENCE = 30

    # Numerical stability
    CLAMP_LOGVAR = (-10.0, 10.0)  # Clamp log-variance for stability
    EPS = 1e-8  # Small epsilon for numerical stability

    # Validation
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1  # From training set

    # Paths
    DATA_PATH = "F:\Abaqus_Mu\MSE\MSE_TASK_2.csv"  # TODO: 修改为你的数据路径
    MODEL_SAVE_PATH = "F:\Abaqus_Mu\MSE/results_2/best_mes_model.pt"
    RESULTS_DIR = "F:\Abaqus_Mu\MSE/results_2"


class HyperparameterSpace:
    """超参数搜索空间定义"""

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> Dict:
        """定义需要优化的超参数"""
        return {
            # 模型架构
            'n_rbf': trial.suggest_int('n_rbf', 30, 100, step=10),
            'rbf_lengthscale': trial.suggest_float('rbf_lengthscale', 0.5, 2.0),
            'hidden_dim1': trial.suggest_categorical('hidden_dim1', [64, 128, 256]),
            'hidden_dim2': trial.suggest_categorical('hidden_dim2', [32, 64, 128]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.3),

            # 训练策略
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-4),
            # 'lambda_phys': trial.suggest_loguniform('lambda_phys', 1e-4, 1e-1),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),

            # 早停策略
            'early_stopping_patience': trial.suggest_int('early_stopping_patience', 20, 50, step=5)
        }


# Initialize
cfg = Config()
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
random.seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)

# Create results directory
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)


# =============================================================================
# 2. DATA LOADING & PREPROCESSING
# =============================================================================

class TabularDataset(Dataset):
    """Custom dataset for tabular data"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_preprocess_data(data_path: str, cfg: Config) -> Tuple:
    """
    Load and preprocess data with log transformations and standardization

    Returns:
        train_loader, val_loader, test_loader, meta_dict
    """
    print("=" * 60)
    print("Loading and preprocessing data...")
    print("=" * 60)

    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")

    # Verify columns
    required_cols = cfg.FEATURE_NAMES + cfg.TARGET_NAMES
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")

    # Check for missing values
    if df[required_cols].isnull().any().any():
        print("Warning: Missing values detected. Dropping rows...")
        df = df.dropna(subset=required_cols)
        print(f"Data shape after dropping NaNs: {df.shape}")

    # Log-transform specified features
    print(f"\nApplying log transformation to: {cfg.LOG_FEATURES}")
    for col in cfg.LOG_FEATURES:
        if col in df.columns:
            if (df[col] <= 0).any():
                print(f"Warning: Non-positive values in {col}, adding offset")
                df[col] = df[col] - df[col].min() + 1e-6
            df[col] = np.log(df[col])

    # Log-transform targets (assume positive)
    print(f"Applying log transformation to targets: {cfg.TARGET_NAMES}")
    for col in cfg.TARGET_NAMES:
        if (df[col] <= 0).any():
            print(f"Warning: Non-positive values in target {col}")
            df[col] = df[col] - df[col].min() + 1e-6
        df[col] = np.log(df[col])

    # Extract arrays
    X = df[cfg.FEATURE_NAMES].values
    y = df[cfg.TARGET_NAMES].values

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")

    # Split: train+val / test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.SEED
    )

    # Further split train+val into train / val
    val_size_adjusted = cfg.VAL_SIZE / (1 - cfg.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_adjusted, random_state=cfg.SEED
    )

    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    # Fit scalers on training data
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    # Transform all sets
    X_train_scaled = x_scaler.transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    X_test_scaled = x_scaler.transform(X_test)

    y_train_scaled = y_scaler.transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)

    # Create datasets and loaders
    train_dataset = TabularDataset(X_train_scaled, y_train_scaled)
    val_dataset = TabularDataset(X_val_scaled, y_val_scaled)
    test_dataset = TabularDataset(X_test_scaled, y_test_scaled)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False
    )

    # Store metadata
    meta = {
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train_scaled': y_train_scaled,
        'y_test_scaled': y_test_scaled,
        'feature_names': cfg.FEATURE_NAMES,
        'target_names': cfg.TARGET_NAMES,
        'log_features': cfg.LOG_FEATURES,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test)
    }

    print("\nData preprocessing completed!")
    return train_loader, val_loader, test_loader, meta


# =============================================================================
# 3. RBF KERNEL FEATURES
# =============================================================================

def fit_rbf_centers(X_scaled: np.ndarray, n_centers: int, seed: int) -> np.ndarray:
    """Fit RBF centers using K-means clustering"""
    print(f"\nFitting {n_centers} RBF centers using K-means...")
    kmeans = KMeans(n_clusters=n_centers, random_state=seed, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)
    centers = kmeans.cluster_centers_
    print(f"RBF centers shape: {centers.shape}")
    return centers


def compute_rbf_features(x: torch.Tensor, centers: torch.Tensor, lengthscale: float) -> torch.Tensor:
    """
    Compute RBF (Gaussian) kernel features

    Args:
        x: Input tensor (B, D)
        centers: RBF centers (C, D)
        lengthscale: Kernel lengthscale parameter

    Returns:
        RBF features (B, C)
    """
    # Compute squared Euclidean distances: (B, 1, D) - (1, C, D) -> (B, C, D)
    diff = x.unsqueeze(1) - centers.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=2)  # (B, C)

    # Apply Gaussian kernel
    rbf = torch.exp(-0.5 * dist_sq / (lengthscale ** 2))
    return rbf


# =============================================================================
# 4. MODEL ARCHITECTURE
# =============================================================================

class MESModel(nn.Module):
    """
    Multi-output Heteroscedastic Neural Network with Physics-Informed Constraints

    Architecture:
        Input -> [Original Features + RBF Features] -> Encoder -> [Mu Head, LogVar Head]
    """

    def __init__(
            self,
            input_dim: int,
            n_rbf: int,
            rbf_lengthscale: float,
            n_outputs: int,
            hidden_dims: List[int],
            dropout_rate: float = 0.1
    ):
        super(MESModel, self).__init__()

        self.input_dim = input_dim
        self.n_rbf = n_rbf
        self.rbf_lengthscale = rbf_lengthscale
        self.n_outputs = n_outputs

        # Total feature dimension: original + RBF
        feature_dim = input_dim + n_rbf

        # Shared encoder
        layers = []
        prev_dim = feature_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output heads
        self.mu_head = nn.Linear(prev_dim, n_outputs)
        self.logvar_head = nn.Linear(prev_dim, n_outputs)

        # Initialize logvar head with small negative bias
        # (so initial variance is not too large)
        nn.init.constant_(self.logvar_head.bias, -1.0)
        nn.init.xavier_uniform_(self.logvar_head.weight, gain=0.01)

    def forward(self, x_scaled: torch.Tensor, centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x_scaled: Scaled input features (B, D)
            centers: RBF centers (C, D)

        Returns:
            mu: Predicted mean (B, n_outputs)
            logvar: Predicted log-variance (B, n_outputs)
        """
        # Compute RBF features
        rbf_features = compute_rbf_features(x_scaled, centers, self.rbf_lengthscale)

        # Concatenate original and RBF features
        features = torch.cat([x_scaled, rbf_features], dim=1)

        # Encode
        encoded = self.encoder(features)

        # Predict mean and log-variance
        mu = self.mu_head(encoded)
        logvar = self.logvar_head(encoded)

        return mu, logvar


# =============================================================================
# 5. LOSS FUNCTIONS
# =============================================================================

def heteroscedastic_nll_loss(
        y_true: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        clamp_range: Tuple[float, float],
        eps: float = 1e-8
) -> torch.Tensor:
    """
    Heteroscedastic Negative Log-Likelihood loss

    Assumes Gaussian likelihood with learned variance:
        p(y|x) = N(mu(x), sigma^2(x))

    NLL = 0.5 * [log(2π) + log(σ²) + (y - μ)² / σ²]
    """
    # Clamp log-variance for numerical stability
    logvar = torch.clamp(logvar, min=clamp_range[0], max=clamp_range[1])
    var = torch.exp(logvar) + eps

    # Compute NLL
    nll = 0.5 * (
            math.log(2 * math.pi) +
            logvar +
            ((y_true - mu) ** 2) / var
    )

    return nll.mean()


def physical_monotonicity_penalty(
        mu: torch.Tensor,
        x: torch.Tensor,
        feature_names: List[str],
        monotonicity_map: Dict[str, Optional[int]],
        device: torch.device
) -> torch.Tensor:
    """
    计算物理单调性惩罚 (L1 强力版)
    """
    penalty = torch.tensor(0.0, device=device, requires_grad=True)

    if not x.requires_grad:
        x = x.requires_grad_(True)

    n_outputs = mu.shape[1]

    for out_idx in range(n_outputs):
        mu_out = mu[:, out_idx]

        # 这里的 create_graph=True 必须保留
        grads = torch.autograd.grad(
            outputs=mu_out.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]

        if grads is None:
            continue

        for feat_idx, feat_name in enumerate(feature_names):
            expected_sign = monotonicity_map.get(feat_name, 0)

            if expected_sign is None or expected_sign == 0:
                continue

            grad_feat = grads[:, feat_idx]

            if expected_sign > 0:
                # 期望正单调，惩罚负梯度
                violation = F.relu(-grad_feat)
            else:
                # 期望负单调，惩罚正梯度
                violation = F.relu(grad_feat)

            
            # 这对修正微小的 RBF 震荡至关重要
            penalty = penalty + violation.mean()

    return penalty


# =============================================================================
# 6. TRAINING & EVALUATION
# =============================================================================

def train_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        centers: torch.Tensor,
        cfg: Config,
        epoch: int
) -> Dict[str, float]:
    """Train for one epoch with Collocation Points"""
    model.train()

    total_nll = 0.0
    total_phys_penalty = 0.0
    total_loss = 0.0
    n_batches = 0

    # 获取特征空间的大致范围，用于随机采样
    # 这里假设输入已经 Standardized (均值0方差1)，所以范围设为 -3 到 3 覆盖绝大多数区域
    # 如果你的数据没有归一化，请根据实际 min/max 修改这里
    sample_min, sample_max = -3.0, 3.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(cfg.DEVICE)
        y_batch = y_batch.to(cfg.DEVICE)

        # 1. 准备训练数据 (用于 NLL Loss)
        x_batch_req = x_batch.clone().detach().requires_grad_(True)

        # -------------------------------------------------------
        # 【关键增强】：生成随机采样点 (Collocation Points)
        # 这一步是为了解决 RBF 模型在训练点之间乱飘、导致单调性只有 29% 的问题
        # -------------------------------------------------------
        batch_size = x_batch.shape[0]
        x_collocation = (sample_max - sample_min) * torch.rand_like(x_batch) + sample_min
        x_collocation = x_collocation.to(cfg.DEVICE).requires_grad_(True)

        # 拼接：物理检查需要覆盖 "训练点" + "未知区域点"
        x_physics_check = torch.cat([x_batch_req, x_collocation], dim=0)
        # -------------------------------------------------------

        optimizer.zero_grad()

        # 2. 前向传播 (针对真实数据计算 NLL)
        mu_pred, logvar_pred = model(x_batch_req, centers)

        nll = heteroscedastic_nll_loss(
            y_batch, mu_pred, logvar_pred,
            clamp_range=cfg.CLAMP_LOGVAR,
            eps=cfg.EPS
        )

        # 3. 前向传播 (针对混合数据计算物理梯度)
        # 注意：这里需要再次调用模型，但这对于强制物理约束是必须的
        mu_phys, _ = model(x_physics_check, centers)

        # 计算物理惩罚
        phys_penalty = physical_monotonicity_penalty(
            mu_phys,
            x_physics_check,  # 使用混合数据
            cfg.FEATURE_NAMES,
            cfg.MONOTONICITY_EXPECTATION,
            cfg.DEVICE
        )

        # 4. 总 Loss
        # 建议：确保 cfg.LAMBDA_PHYS 足够大 (例如 100.0 或 500.0)
        loss = nll + cfg.LAMBDA_PHYS * phys_penalty
        if n_batches == 0:
            print(
                f"DEBUG Epoch {epoch}: NLL={nll.item():.4f}, Phys(Raw)={phys_penalty.item():.6f}, Weighted_Phys={cfg.LAMBDA_PHYS * phys_penalty.item():.4f}")

        # 反向传播
        loss.backward()

        # 梯度裁剪 (防止 RBF 梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 记录指标
        total_nll += nll.item()
        total_phys_penalty += phys_penalty.item()
        total_loss += loss.item()
        n_batches += 1

    metrics = {
        'train_nll': total_nll / n_batches,
        'train_phys_penalty': total_phys_penalty / n_batches,
        'train_loss': total_loss / n_batches
    }

    return metrics


@torch.no_grad()
def evaluate(
        model: nn.Module,
        data_loader: DataLoader,
        centers: torch.Tensor,
        cfg: Config
) -> Dict[str, float]:
    """Evaluate model on validation/test set"""
    model.eval()

    total_nll = 0.0
    n_batches = 0

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(cfg.DEVICE)
        y_batch = y_batch.to(cfg.DEVICE)

        mu_pred, logvar_pred = model(x_batch, centers)

        nll = heteroscedastic_nll_loss(
            y_batch, mu_pred, logvar_pred,
            clamp_range=cfg.CLAMP_LOGVAR,
            eps=cfg.EPS
        )

        total_nll += nll.item()
        n_batches += 1

    metrics = {
        'nll': total_nll / n_batches
    }

    return metrics


def train_model_with_config(
        train_loader: DataLoader,
        val_loader: DataLoader,
        meta: Dict,
        cfg: Config,
        hyperparams: Dict = None,
        verbose: bool = True  # 新增：控制是否打印调试信息
) -> Tuple[nn.Module, np.ndarray, Dict, float]:
    """
    带超参数配置的训练函数

    Args:
        verbose: 是否打印详细调试信息

    Returns:
        model, centers, history, best_val_nll
    """

    # 如果提供了超参数，覆盖默认配置
    if hyperparams:
        n_rbf = hyperparams['n_rbf']
        rbf_lengthscale = hyperparams['rbf_lengthscale']
        hidden_dims = [hyperparams['hidden_dim1'], hyperparams['hidden_dim2']]
        dropout_rate = hyperparams['dropout_rate']
        learning_rate = hyperparams['learning_rate']
        weight_decay = hyperparams['weight_decay']
        # lambda_phys = hyperparams['lambda_phys']
        lambda_phys = cfg.LAMBDA_PHYS
        batch_size = hyperparams['batch_size']
        patience = hyperparams['early_stopping_patience']

        if verbose:
            print("\n📝 使用自定义超参数:")
            print(f"  - n_rbf: {n_rbf}")
            print(f"  - rbf_lengthscale: {rbf_lengthscale}")
            print(f"  - hidden_dims: {hidden_dims}")
            print(f"  - dropout_rate: {dropout_rate}")
            print(f"  - learning_rate: {learning_rate}")
            print(f"  - weight_decay: {weight_decay}")
            print(f"  - lambda_phys: {lambda_phys} ⚠️ (物理惩罚权重)")
            print(f"  - batch_size: {batch_size}")
    else:
        # 使用默认配置
        n_rbf = cfg.N_RBF
        rbf_lengthscale = cfg.RBF_LENGTHSCALE
        hidden_dims = cfg.HIDDEN_DIMS
        dropout_rate = cfg.DROPOUT_RATE
        learning_rate = cfg.LEARNING_RATE
        weight_decay = cfg.WEIGHT_DECAY
        lambda_phys = cfg.LAMBDA_PHYS
        batch_size = cfg.BATCH_SIZE
        patience = cfg.EARLY_STOPPING_PATIENCE

        if verbose:
            print("\n📝 使用默认配置")
            print(f"  - lambda_phys: {lambda_phys} ⚠️ (物理惩罚权重)")

    # 如果batch_size改变，重建DataLoader
    if hyperparams and batch_size != cfg.BATCH_SIZE:
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=batch_size,
            shuffle=False
        )

    # Fit RBF centers
    centers_np = fit_rbf_centers(
        meta['X_train_scaled'],
        n_centers=n_rbf,
        seed=cfg.SEED
    )
    centers_torch = torch.FloatTensor(centers_np).to(cfg.DEVICE)

    # Initialize model
    model = MESModel(
        input_dim=len(cfg.FEATURE_NAMES),
        n_rbf=n_rbf,
        rbf_lengthscale=rbf_lengthscale,
        n_outputs=len(cfg.TARGET_NAMES),
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    ).to(cfg.DEVICE)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )

    # Training history (扩展以记录物理惩罚)
    history = {
        'train_nll': [],
        'val_nll': [],
        'train_phys': [],  # 新增：记录物理惩罚
        'train_total': [],  # 新增：记录总损失
        'learning_rate': []  # 新增：记录学习率
    }

    # Early stopping
    best_val_nll = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    if verbose:
        print("\n" + "=" * 80)
        print("🚀 开始训练...")
        print("=" * 80)

    # Training loop
    for epoch in range(cfg.N_EPOCHS):
        # Train
        model.train()
        train_nll_sum = 0.0
        train_phys_sum = 0.0
        train_total_sum = 0.0
        n_batches = 0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(cfg.DEVICE)
            y_batch = y_batch.to(cfg.DEVICE)

            # 训练数据点，用于 NLL
            x_batch_req = x_batch.clone().detach().requires_grad_(True)

            # ===== 新增：生成 collocation 点，用于物理约束 =====
            sample_min, sample_max = -3.0, 3.0  # 对 standardized 特征来说基本覆盖[-3σ,3σ]
            x_collocation = (sample_max - sample_min) * torch.rand_like(x_batch) + sample_min
            x_collocation = x_collocation.to(cfg.DEVICE).requires_grad_(True)

            # 拼接：在更多区域检查单调性
            x_physics = torch.cat([x_batch_req, x_collocation], dim=0)

            optimizer.zero_grad()

            # 1) 真实数据上的 NLL
            mu_pred, logvar_pred = model(x_batch_req, centers_torch)
            nll = heteroscedastic_nll_loss(
                y_batch, mu_pred, logvar_pred,
                clamp_range=cfg.CLAMP_LOGVAR,
                eps=cfg.EPS
            )

            # 2) 真实数据 + collocation 上的物理惩罚
            mu_phys, _ = model(x_physics, centers_torch)
            phys_penalty = physical_monotonicity_penalty(
                mu_phys, x_physics,
                cfg.FEATURE_NAMES,
                cfg.MONOTONICITY_EXPECTATION,
                cfg.DEVICE
            )

            loss = nll + lambda_phys * phys_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_nll_sum += nll.item()
            train_phys_sum += phys_penalty.item()
            train_total_sum += loss.item()
            n_batches += 1

            # 🔍 关键调试：打印第一个batch的详细信息
            if verbose and batch_idx == 0 and (epoch % 10 == 0 or epoch < 5):
                print(f"\n📊 Epoch {epoch + 1} - Batch 1 详细分解:")
                print(f"  NLL Loss:           {nll.item():.6f}")
                print(f"  Physics Penalty:    {phys_penalty.item():.6f}")
                print(f"  Weighted Physics:   {(lambda_phys * phys_penalty).item():.6f}  (λ={lambda_phys})")
                print(f"  Total Loss:         {loss.item():.6f}")
                print(f"  物理惩罚占比:        {(lambda_phys * phys_penalty / loss * 100).item():.2f}%")

                # 检查物理惩罚是否为0或异常小
                if phys_penalty.item() < 1e-8:
                    print("  ⚠️  警告：物理惩罚几乎为0，可能未生效！")

        # 计算epoch平均值
        avg_train_nll = train_nll_sum / n_batches
        avg_train_phys = train_phys_sum / n_batches
        avg_train_total = train_total_sum / n_batches

        # Validate
        val_metrics = evaluate(model, val_loader, centers_torch, cfg)

        # 记录历史
        history['train_nll'].append(avg_train_nll)
        history['val_nll'].append(val_metrics['nll'])
        history['train_phys'].append(avg_train_phys)
        history['train_total'].append(avg_train_total)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # 学习率调度
        scheduler.step(val_metrics['nll'])

        # 🔍 定期打印训练进度
        if verbose and (epoch % 20 == 0 or epoch < 5 or epoch == cfg.N_EPOCHS - 1):
            print(f"\n{'=' * 80}")
            print(f"📈 Epoch {epoch + 1}/{cfg.N_EPOCHS}")
            print(f"{'=' * 80}")
            print(f"  Train NLL:          {avg_train_nll:.6f}")
            print(f"  Train Physics:      {avg_train_phys:.6f}")
            print(f"  Train Total:        {avg_train_total:.6f}")
            print(f"  Val NLL:            {val_metrics['nll']:.6f}")
            print(f"  Learning Rate:      {optimizer.param_groups[0]['lr']:.6e}")
            print(f"  Best Val NLL:       {best_val_nll:.6f}")
            print(f"  No Improve Count:   {epochs_no_improve}/{patience}")

        # Early stopping
        if val_metrics['nll'] < best_val_nll - 1e-6:
            best_val_nll = val_metrics['nll']
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()  # 保存最佳模型
            if verbose:
                print(f"  ✅ 新的最佳模型！Val NLL降低到 {best_val_nll:.6f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            if verbose:
                print(f"\n⏹️  Early stopping triggered at epoch {epoch + 1}")
            break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"\n✅ 已加载最佳模型 (Val NLL: {best_val_nll:.6f})")

    if verbose:
        print("\n" + "=" * 80)
        print("🎉 训练完成!")
        print("=" * 80)
        print(f"最终统计:")
        print(f"  - 总训练轮数: {len(history['train_nll'])}")
        print(f"  - 最佳验证NLL: {best_val_nll:.6f}")
        print(f"  - 最终训练NLL: {history['train_nll'][-1]:.6f}")
        print(f"  - 最终物理惩罚: {history['train_phys'][-1]:.6f}")

        # 🔍 最终诊断
        final_phys_penalty = history['train_phys'][-1]
        if final_phys_penalty < 1e-6:
            print(f"\n⚠️  警告：最终物理惩罚={final_phys_penalty:.8f}，几乎为0！")
            print(f"   建议检查：")
            print(f"   1. physical_monotonicity_penalty() 函数是否正确返回值")
            print(f"   2. MONOTONICITY_EXPECTATION 是否有定义约束特征")
            print(f"   3. lambda_phys={lambda_phys} 是否太小")

    return model, centers_np, history, best_val_nll


# =============================================================================
# 7. PREDICTION & POST-PROCESSING
# =============================================================================

@torch.no_grad()
def predict_on_loader(
        model: nn.Module,
        data_loader: DataLoader,
        centers: torch.Tensor,
        cfg: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions on a dataset

    Returns:
        pred_mu: Predicted means (N, n_outputs)
        pred_logvar: Predicted log-variances (N, n_outputs)
        y_true: True targets (N, n_outputs)
    """
    model.eval()

    pred_mu_list = []
    pred_logvar_list = []
    y_true_list = []

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(cfg.DEVICE)
        y_batch = y_batch.to(cfg.DEVICE)

        mu, logvar = model(x_batch, centers)

        pred_mu_list.append(mu.cpu().numpy())
        pred_logvar_list.append(logvar.cpu().numpy())
        y_true_list.append(y_batch.cpu().numpy())

    pred_mu = np.vstack(pred_mu_list)
    pred_logvar = np.vstack(pred_logvar_list)
    y_true = np.vstack(y_true_list)

    return pred_mu, pred_logvar, y_true


def transform_to_original_space(
        pred_mu_scaled: np.ndarray,
        pred_logvar_scaled: np.ndarray,
        y_scaler: StandardScaler
) -> Dict[str, np.ndarray]:
    """
    Transform predictions from scaled log-space back to original space

    Model predicts in scaled y' space where y' = ln(y)
    Need to:
        1. Unscale to get y' (log-space)
        2. Apply lognormal transformation to get original space statistics

    For Y ~ LogNormal(μ', σ'²):
        E[Y] = exp(μ' + σ'²/2)
        Var[Y] = exp(2μ' + σ'²) * (exp(σ'²) - 1)
    """
    # Unscale predictions to log-space
    mu_logspace = y_scaler.inverse_transform(pred_mu_scaled)

    # Adjust log-variance for scaling
    # If y_scaled = (y - mean) / std, then:
    # Var(y_scaled) = Var(y) / std²
    # logvar_scaled = log(Var(y) / std²) = logvar_y - 2*log(std)
    # Therefore: logvar_y = logvar_scaled + 2*log(std)
    std_y = y_scaler.scale_  # Shape: (n_outputs,)
    pred_logvar_logspace = pred_logvar_scaled + 2.0 * np.log(std_y[None, :])

    # Compute variance in log-space
    sigma2_logspace = np.exp(pred_logvar_logspace)

    # Transform to original space using lognormal formulas
    E_Y = np.exp(mu_logspace + 0.5 * sigma2_logspace)
    Var_Y = np.exp(2 * mu_logspace + sigma2_logspace) * (np.exp(sigma2_logspace) - 1.0)
    Std_Y = np.sqrt(Var_Y)

    # 95% confidence intervals in original space
    # For lognormal: CI = exp(μ' ± 1.96*σ')
    sigma_logspace = np.sqrt(sigma2_logspace)
    CI_lower = np.exp(mu_logspace - 1.96 * sigma_logspace)
    CI_upper = np.exp(mu_logspace + 1.96 * sigma_logspace)

    return {
        'mu_logspace': mu_logspace,
        'sigma2_logspace': sigma2_logspace,
        'E_Y': E_Y,
        'Var_Y': Var_Y,
        'Std_Y': Std_Y,
        'CI_lower': CI_lower,
        'CI_upper': CI_upper
    }


# =============================================================================
# 8. EVALUATION METRICS & DIAGNOSTICS
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, output_names: List[str]) -> pd.DataFrame:
    """
    Compute comprehensive evaluation metrics for each output
    """
    n_outputs = y_true.shape[1]
    metrics_list = []

    for i in range(n_outputs):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae = mean_absolute_error(y_t, y_p)
        r2 = r2_score(y_t, y_p)

        # Mean absolute percentage error
        mape = np.mean(np.abs((y_t - y_p) / (y_t + 1e-8))) * 100

        # Correlation coefficient
        corr = np.corrcoef(y_t, y_p)[0, 1]

        metrics_list.append({
            'Output': output_names[i],
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE (%)': mape,
            'Correlation': corr
        })

    return pd.DataFrame(metrics_list)


def check_monotonicity(
        model: nn.Module,
        X_scaled: np.ndarray,
        centers: torch.Tensor,
        feature_names: List[str],
        monotonicity_map: Dict[str, Optional[int]],
        cfg: Config,
        n_samples: int = 100
) -> pd.DataFrame:
    """
    Check monotonicity compliance on a sample of data
    """
    model.eval()

    # Random sample
    indices = np.random.choice(len(X_scaled), size=min(n_samples, len(X_scaled)), replace=False)
    X_sample = torch.FloatTensor(X_scaled[indices]).to(cfg.DEVICE)
    X_sample.requires_grad_(True)

    # Forward pass
    mu, _ = model(X_sample, centers)

    # Compute gradients for each output
    results = []
    n_outputs = mu.shape[1]

    for out_idx in range(n_outputs):
        mu_out = mu[:, out_idx]
        grads = grad(
            outputs=mu_out.sum(),
            inputs=X_sample,
            create_graph=False,
            retain_graph=True
        )[0].cpu().numpy()  # (n_samples, n_features)

        # Check each feature
        for feat_idx, feat_name in enumerate(feature_names):
            expected_sign = monotonicity_map.get(feat_name, None)

            if expected_sign is None:
                continue

            grad_feat = grads[:, feat_idx]

            if expected_sign > 0:
                compliance = (grad_feat >= 0).mean() * 100
                expected = "Increasing"
            else:
                compliance = (grad_feat <= 0).mean() * 100
                expected = "Decreasing"

            results.append({
                'Output': cfg.TARGET_NAMES[out_idx],
                'Feature': feat_name,
                'Expected': expected,
                'Compliance (%)': compliance,
                'Mean Gradient': grad_feat.mean(),
                'Std Gradient': grad_feat.std()
            })

    return pd.DataFrame(results)


def residual_diagnostics(
        y_true_logspace: np.ndarray,
        y_pred_logspace: np.ndarray,
        output_names: List[str],
        save_dir: str
):
    """
    Generate residual diagnostic plots
    """
    n_outputs = y_true_logspace.shape[1]

    fig, axes = plt.subplots(n_outputs, 3, figsize=(15, 5 * n_outputs))
    if n_outputs == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_outputs):
        y_t = y_true_logspace[:, i]
        y_p = y_pred_logspace[:, i]
        residuals = y_t - y_p

        # Residual vs Predicted
        axes[i, 0].scatter(y_p, residuals, alpha=0.5, s=20)
        axes[i, 0].axhline(y=0, color='r', linestyle='--')
        axes[i, 0].set_xlabel('Predicted (log-space)')
        axes[i, 0].set_ylabel('Residuals')
        axes[i, 0].set_title(f'{output_names[i]}: Residuals vs Predicted')
        axes[i, 0].grid(True, alpha=0.3)

        # Histogram
        axes[i, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[i, 1].set_xlabel('Residuals')
        axes[i, 1].set_ylabel('Frequency')
        axes[i, 1].set_title(f'{output_names[i]}: Residual Distribution')
        axes[i, 1].grid(True, alpha=0.3)

        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[i, 2])
        axes[i, 2].set_title(f'{output_names[i]}: Q-Q Plot')
        axes[i, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'residual_diagnostics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Residual diagnostics saved to {save_dir}")


# =============================================================================
# 9. VISUALIZATION
# =============================================================================

def plot_training_history(history: Dict, save_dir: str):
    """
    绘制训练历史（兼容旧版和新版history格式）

    新版包含：train_nll, val_nll, train_phys, train_total, learning_rate
    旧版包含：train_nll, val_nll
    """

    # 检查是否有新增字段
    has_physics = 'train_phys' in history and len(history['train_phys']) > 0
    has_total = 'train_total' in history and len(history['train_total']) > 0
    has_lr = 'learning_rate' in history and len(history['learning_rate']) > 0

    # 根据可用数据决定子图布局
    if has_physics and has_total and has_lr:
        # 完整版：2×2布局
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()  # 展平为一维数组，方便索引

        # 1. NLL损失
        axes[0].plot(history['train_nll'], label='Train NLL', linewidth=2)
        axes[0].plot(history['val_nll'], label='Val NLL', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('NLL Loss', fontsize=12)
        axes[0].set_title('Negative Log-Likelihood', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. 总损失
        axes[1].plot(history['train_total'], label='Total Loss', color='red', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Total Loss', fontsize=12)
        axes[1].set_title('Total Loss (NLL + λ×Physics)', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. 物理惩罚（关键！）
        axes[2].plot(history['train_phys'], label='Physics Penalty', color='orange', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Physics Penalty', fontsize=12)
        axes[2].set_title('Physical Monotonicity Penalty', fontsize=13, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # 添加警告线
        mean_phys = np.mean(history['train_phys'])
        if mean_phys < 1e-4:
            axes[2].axhline(y=1e-4, color='r', linestyle='--', linewidth=1, alpha=0.7)
            axes[2].text(0.5, 1e-4, '⚠️ 物理惩罚过低',
                         transform=axes[2].get_yaxis_transform(),
                         ha='left', va='bottom', fontsize=9, color='red')

        # 4. 学习率
        axes[3].plot(history['learning_rate'], label='Learning Rate', color='green', linewidth=2)
        axes[3].set_xlabel('Epoch', fontsize=12)
        axes[3].set_ylabel('Learning Rate', fontsize=12)
        axes[3].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        axes[3].set_yscale('log')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        filename = 'training_history_enhanced.png'

    else:
        # 简化版：只有NLL损失
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.plot(history['train_nll'], label='Train NLL', linewidth=2, marker='o', markersize=3)
        ax.plot(history['val_nll'], label='Val NLL', linewidth=2, marker='s', markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('NLL Loss', fontsize=12)
        ax.set_title('Training History - Negative Log-Likelihood', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # 标注最佳点
        best_epoch = np.argmin(history['val_nll'])
        best_val_nll = history['val_nll'][best_epoch]
        ax.scatter([best_epoch], [best_val_nll], color='red', s=100, zorder=5,
                   label=f'Best (Epoch {best_epoch + 1})')
        ax.legend(fontsize=11)

        filename = 'training_history.png'

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Training history plot saved to: {save_path}")

    # 打印关键统计信息
    print("\n📊 训练历史统计:")
    print(f"  - 训练轮数: {len(history['train_nll'])}")
    print(f"  - 最佳验证NLL: {min(history['val_nll']):.6f} (Epoch {np.argmin(history['val_nll']) + 1})")
    print(f"  - 最终训练NLL: {history['train_nll'][-1]:.6f}")

    if has_physics:
        final_phys = history['train_phys'][-1]
        print(f"  - 最终物理惩罚: {final_phys:.6f}")

        if final_phys < 1e-6:
            print(f"\n  ⚠️  警告：物理惩罚过低 ({final_phys:.8f})")
            print(f"     建议检查:")
            print(f"     1. MONOTONICITY_EXPECTATION 是否正确定义")
            print(f"     2. lambda_phys 是否需要增大")
        elif final_phys > 0.01:
            print(f"\n  ✅ 物理惩罚正常 ({final_phys:.6f})")


def plot_predictions(
        y_true_orig: np.ndarray,
        predictions: Dict[str, np.ndarray],
        output_names: List[str],
        save_dir: str
):
    """
    Plot predicted vs true values with uncertainty
    """
    n_outputs = y_true_orig.shape[1]

    fig, axes = plt.subplots(1, n_outputs, figsize=(7 * n_outputs, 6))
    if n_outputs == 1:
        axes = [axes]

    for i in range(n_outputs):
        y_true = y_true_orig[:, i]
        y_pred_mean = predictions['E_Y'][:, i]
        y_pred_std = predictions['Std_Y'][:, i]
        ci_lower = predictions['CI_lower'][:, i]
        ci_upper = predictions['CI_upper'][:, i]

        # Scatter plot with error bars
        axes[i].errorbar(
            y_true, y_pred_mean,
            yerr=1.96 * y_pred_std,
            fmt='o', alpha=0.5, markersize=4,
            elinewidth=1, capsize=2,
            label='Predictions ± 1.96σ'
        )

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred_mean.min())
        max_val = max(y_true.max(), y_pred_mean.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')

        # Compute R²
        r2 = r2_score(y_true, y_pred_mean)

        axes[i].set_xlabel(f'True {output_names[i]}', fontsize=12)
        axes[i].set_ylabel(f'Predicted {output_names[i]}', fontsize=12)
        axes[i].set_title(f'{output_names[i]}\n(R² = {r2:.4f})', fontsize=13)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions_vs_true.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Prediction plot saved to {save_dir}")


def plot_uncertainty_distribution(
        predictions: Dict[str, np.ndarray],
        output_names: List[str],
        save_dir: str
):
    """
    Plot distribution of predicted uncertainties
    """
    n_outputs = len(output_names)

    fig, axes = plt.subplots(1, n_outputs, figsize=(7 * n_outputs, 5))
    if n_outputs == 1:
        axes = [axes]

    for i in range(n_outputs):
        std_y = predictions['Std_Y'][:, i]
        mean_y = predictions['E_Y'][:, i]

        # Coefficient of variation
        cv = std_y / (mean_y + 1e-8) * 100

        axes[i].hist(cv, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[i].axvline(cv.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {cv.mean():.2f}%')
        axes[i].set_xlabel('Coefficient of Variation (%)', fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)
        axes[i].set_title(f'{output_names[i]}: Predicted Uncertainty', fontsize=13)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'uncertainty_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Uncertainty distribution plot saved to {save_dir}")


def plot_partial_dependence(
        model: nn.Module,
        centers: torch.Tensor,
        X_scaled: np.ndarray,
        x_scaler: StandardScaler,
        feature_names: List[str],
        output_names: List[str],
        cfg: Config,
        save_dir: str,
        n_features_to_plot: int = 6
):
    """
    Plot partial dependence plots for key features
    """
    model.eval()

    # Select features to plot (those with monotonicity constraints)
    features_to_plot = [
                           f for f in feature_names
                           if cfg.MONOTONICITY_EXPECTATION.get(f, None) is not None
                       ][:n_features_to_plot]

    n_features = len(features_to_plot)
    n_outputs = len(output_names)

    fig, axes = plt.subplots(n_outputs, n_features, figsize=(4 * n_features, 4 * n_outputs))
    if n_outputs == 1:
        axes = axes.reshape(1, -1)
    if n_features == 1:
        axes = axes.reshape(-1, 1)

    # Use median of other features
    X_median = np.median(X_scaled, axis=0)

    for feat_idx, feat_name in enumerate(features_to_plot):
        # Get feature index
        f_idx = feature_names.index(feat_name)

        # Create range for this feature
        feat_min = X_scaled[:, f_idx].min()
        feat_max = X_scaled[:, f_idx].max()
        feat_range = np.linspace(feat_min, feat_max, 100)

        # Create input matrix
        X_pdp = np.tile(X_median, (100, 1))
        X_pdp[:, f_idx] = feat_range

        # Predict
        X_pdp_torch = torch.FloatTensor(X_pdp).to(cfg.DEVICE)
        with torch.no_grad():
            mu_pred, logvar_pred = model(X_pdp_torch, centers)
            mu_pred = mu_pred.cpu().numpy()
            logvar_pred = logvar_pred.cpu().numpy()

        # Transform back to original feature scale (for x-axis)
        X_pdp_orig = x_scaler.inverse_transform(X_pdp)
        feat_range_orig = X_pdp_orig[:, f_idx]

        # If feature was log-transformed, exp it back
        if feat_name in cfg.LOG_FEATURES:
            feat_range_orig = np.exp(feat_range_orig)

        # Plot for each output
        for out_idx in range(n_outputs):
            mu_out = mu_pred[:, out_idx]
            std_out = np.sqrt(np.exp(logvar_pred[:, out_idx]))

            axes[out_idx, feat_idx].plot(feat_range_orig, mu_out, 'b-', linewidth=2, label='Mean')
            axes[out_idx, feat_idx].fill_between(
                feat_range_orig,
                mu_out - 1.96 * std_out,
                mu_out + 1.96 * std_out,
                alpha=0.3, label='95% CI'
            )

            axes[out_idx, feat_idx].set_xlabel(feat_name, fontsize=11)
            if feat_idx == 0:
                axes[out_idx, feat_idx].set_ylabel(f'{output_names[out_idx]}\n(log-space)', fontsize=11)
            axes[out_idx, feat_idx].grid(True, alpha=0.3)
            if out_idx == 0 and feat_idx == 0:
                axes[out_idx, feat_idx].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'partial_dependence.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Partial dependence plots saved to {save_dir}")


def objective(
        trial: optuna.Trial,
        train_loader: DataLoader,
        val_loader: DataLoader,
        meta: Dict,
        cfg: Config
) -> float:
    """
    Optuna优化目标函数

    Args:
        trial: Optuna trial对象
        train_loader, val_loader: 数据加载器
        meta: 元数据
        cfg: 基础配置

    Returns:
        验证集NLL (越小越好)
    """

    # 获取本次trial的超参数
    hyperparams = HyperparameterSpace.suggest_params(trial)

    try:
        # 训练模型
        model, centers, history, best_val_nll = train_model_with_config(
            train_loader, val_loader, meta, cfg, hyperparams
        )

        # 报告中间结果（用于剪枝）
        for epoch, val_nll in enumerate(history['val_nll']):
            trial.report(val_nll, epoch)

            # 如果效果不好，提前终止
            if trial.should_prune():
                raise optuna.TrialPruned()

        # 记录额外信息
        trial.set_user_attr('final_train_nll', history['train_nll'][-1])
        trial.set_user_attr('n_epochs', len(history['train_nll']))

        return best_val_nll

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float('inf')


def run_hyperparameter_tuning(
        train_loader: DataLoader,
        val_loader: DataLoader,
        meta: Dict,
        cfg: Config,
        n_trials: int = 50,
        timeout: int = 3600  # 1小时
) -> optuna.Study:
    """
    运行超参数调优

    Args:
        n_trials: 最多尝试多少组参数
        timeout: 最长运行时间(秒)

    Returns:
        Optuna Study对象
    """

    print("\n" + "=" * 80)
    print("🚀 启动超参数自动调优")
    print("=" * 80)

    # 创建Study（使用TPE采样器）
    study = optuna.create_study(
        study_name="mes_hyperparameter_tuning",
        direction='minimize',  # 最小化验证NLL
        sampler=optuna.samplers.TPESampler(seed=cfg.SEED),
        pruner=optuna.pruners.MedianPruner(  # 剪枝不佳的trial
            n_startup_trials=5,
            n_warmup_steps=10
        )
    )

    # 运行优化
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, meta, cfg),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        callbacks=[
            lambda study, trial: print(
                f"Trial {trial.number} finished | "
                f"Val NLL: {trial.value:.4f} | "
                f"Best so far: {study.best_value:.4f}"
            )
        ]
    )

    # 打印结果
    print("\n" + "=" * 80)
    print("🏆 调优完成！")
    print("=" * 80)
    print(f"\n最佳验证NLL: {study.best_value:.4f}")
    print(f"\n最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 保存结果
    study.trials_dataframe().to_csv(
        os.path.join(cfg.RESULTS_DIR, 'optuna_trials.csv'),
        index=False
    )

    # 生成可视化
    try:
        import optuna.visualization as vis

        # 参数重要性
        fig = vis.plot_param_importances(study)
        fig.write_html(os.path.join(cfg.RESULTS_DIR, 'param_importance.html'))

        # 优化历史
        fig = vis.plot_optimization_history(study)
        fig.write_html(os.path.join(cfg.RESULTS_DIR, 'optimization_history.html'))

        # 参数关系
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(cfg.RESULTS_DIR, 'parallel_coordinate.html'))

        print(f"\n📊 可视化结果已保存到: {cfg.RESULTS_DIR}/")
    except ImportError:
        print("\n提示: 安装 plotly 可生成交互式图表: pip install plotly")

    return study


def export_for_reliability_analysis(model, X_train, X_test, y_train, y_test, meta, save_dir='results'):
    """
    导出数据用于可靠性分析 - 兼容 reliability_analysis_full.py

    Args:
        model: 训练好的MESModel
        X_train: 训练集特征 (已标准化)
        X_test: 测试集特征 (已标准化)
        y_train: 训练集标签 (已标准化)
        y_test: 测试集标签 (已标准化)
        meta: 包含centers等元数据的字典
        save_dir: 保存目录
    """
    import os
    import numpy as np
    import torch
    import joblib

    print("\n" + "=" * 80)
    print("📤 导出数据用于可靠性分析")
    print("=" * 80)

    # 创建输出目录
    export_dir = os.path.join(save_dir, 'results_2')  # 匹配可靠性分析代码的路径
    os.makedirs(export_dir, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    # ============================================================
    # 1. 获取RBF centers
    # ============================================================
    if 'centers' not in meta:
        print("⚠️ Centers not found in meta, recomputing...")
        if hasattr(model, 'n_rbf') and model.n_rbf > 0:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=model.n_rbf, random_state=42, n_init=10)
            kmeans.fit(X_train)
            meta['centers'] = kmeans.cluster_centers_
            print(f"   ✓ Recomputed {model.n_rbf} RBF centers")
        else:
            meta['centers'] = np.zeros((1, X_train.shape[1]))
            print("   ℹ️ Model has no RBF layer")

    centers = torch.FloatTensor(meta['centers']).to(device)
    print(f"✓ Using {centers.shape[0]} RBF centers with shape {centers.shape}")

    # ============================================================
    # 2. 训练集预测
    # ============================================================
    print("\n1️⃣ Making predictions on training set...")
    X_train_t = torch.FloatTensor(X_train).to(device)

    with torch.no_grad():
        pred_mu_train, pred_logvar_train = model(X_train_t, centers)

    pred_mu_train = pred_mu_train.cpu().numpy()
    pred_sigma_train = np.sqrt(np.exp(pred_logvar_train.cpu().numpy()))
    print(f"   ✓ Training predictions: μ shape {pred_mu_train.shape}, σ shape {pred_sigma_train.shape}")

    # ============================================================
    # 3. 测试集预测
    # ============================================================
    print("\n2️⃣ Making predictions on test set...")
    X_test_t = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        pred_mu_test, pred_logvar_test = model(X_test_t, centers)

    pred_mu_test = pred_mu_test.cpu().numpy()
    pred_sigma_test = np.sqrt(np.exp(pred_logvar_test.cpu().numpy()))
    print(f"   ✓ Test predictions: μ shape {pred_mu_test.shape}, σ shape {pred_sigma_test.shape}")

    # ============================================================
    # 4. 保存为可靠性分析所需格式
    # ============================================================
    print("\n3️⃣ Saving data in reliability analysis format...")

    # 4.1 MES预测结果 (mes_predictions.pkl)
    mes_predictions = {
        'X_train_scaled': X_train,
        'y_train_scaled': y_train,
        'pred_mu': pred_mu_train,
        'pred_sigma': pred_sigma_train,
        # 可选：包含测试集
        'X_test_scaled': X_test,
        'y_test_scaled': y_test,
        'pred_mu_test': pred_mu_test,
        'pred_sigma_test': pred_sigma_test
    }
    joblib.dump(mes_predictions, os.path.join(export_dir, 'mes_predictions.pkl'))
    print(f"   ✓ Saved mes_predictions.pkl")

    # 4.2 预处理数据 (preprocessed_data.pkl)
    preprocessed_data = {
        'X_scaler': meta.get('x_scaler'),  # 小写 x
        'y_scaler': meta.get('y_scaler'),
        'feature_names': meta.get('feature_names', [f'X{i}' for i in range(X_train.shape[1])]),
        'output_names': meta.get('target_names', ['y1', 'y2']),
        'centers': meta['centers'],
        'n_rbf': model.n_rbf if hasattr(model, 'n_rbf') else 0
    }

    joblib.dump(preprocessed_data, os.path.join(export_dir, 'preprocessed_data.pkl'))
    print(f"   ✓ Saved preprocessed_data.pkl")

    # ============================================================
    # 5. 计算统计信息
    # ============================================================
    print("\n4️⃣ Computing statistics...")

    # 训练集指标
    train_mae = np.mean(np.abs(y_train - pred_mu_train), axis=0)
    train_rmse = np.sqrt(np.mean((y_train - pred_mu_train) ** 2, axis=0))
    train_r2 = [np.corrcoef(y_train[:, i], pred_mu_train[:, i])[0, 1] ** 2
                for i in range(y_train.shape[1])]

    # 测试集指标
    test_mae = np.mean(np.abs(y_test - pred_mu_test), axis=0)
    test_rmse = np.sqrt(np.mean((y_test - pred_mu_test) ** 2, axis=0))
    test_r2 = [np.corrcoef(y_test[:, i], pred_mu_test[:, i])[0, 1] ** 2
               for i in range(y_test.shape[1])]

    # ============================================================
    # 6. 打印摘要
    # ============================================================
    print("\n" + "=" * 80)
    print("📊 EXPORT SUMMARY")
    print("=" * 80)

    print(f"\n📁 Output Directory: {export_dir}")
    print(f"   ├── mes_predictions.pkl")
    print(f"   └── preprocessed_data.pkl")

    print(f"\n📐 Data Shapes:")
    print(f"   Training:   X={X_train.shape}, y={y_train.shape}")
    print(f"   Test:       X={X_test.shape}, y={y_test.shape}")
    print(f"   Predictions: μ={pred_mu_train.shape}, σ={pred_sigma_train.shape}")

    print(f"\n🎯 Training Set Metrics:")
    for i, name in enumerate(preprocessed_data['output_names']):
        print(f"   {name}: MAE={train_mae[i]:.4f}, RMSE={train_rmse[i]:.4f}, R²={train_r2[i]:.4f}")

    print(f"\n🎯 Test Set Metrics:")
    for i, name in enumerate(preprocessed_data['output_names']):
        print(f"   {name}: MAE={test_mae[i]:.4f}, RMSE={test_rmse[i]:.4f}, R²={test_r2[i]:.4f}")

    print("\n" + "=" * 80)
    print("✅ Data ready for reliability_analysis_full.py")
    print(f"   Update MES_OUTPUT_DIR to: {os.path.abspath(export_dir)}")
    print("=" * 80)

    return export_dir


# =============================================================================
# 10. MAIN EXECUTION
# =============================================================================

def main():
    """主执行流程"""
    print("\n" + "=" * 80)
    print("PHYSICS-INFORMED HETEROSCEDASTIC NEURAL NETWORK FOR MES PREDICTION")
    print("=" * 80)

    # 检查数据文件
    if not os.path.exists(cfg.DATA_PATH):
        print(f"\nERROR: Data file not found at {cfg.DATA_PATH}")
        return

    # 加载数据
    train_loader, val_loader, test_loader, meta = load_and_preprocess_data(
        cfg.DATA_PATH, cfg
    )

    # ========== 新增：定义超参数保存路径 ==========
    best_params_path = os.path.join(cfg.RESULTS_DIR, 'best_hyperparameters.json')

    # ========== 选择模式 ==========
    print("\n请选择运行模式:")
    print("1. 使用默认超参数训练")
    print("2. 自动调优超参数")
    print("3. 加载已保存的最佳超参数并训练 (推荐)")  # 新增选项
    mode = input("请输入选项 (1/2/3): ").strip()

    best_hyperparams = None

    if mode == '3':
        # ========== 新增：加载已有超参数 ==========
        if not os.path.exists(best_params_path):
            print(f"\n❌ 未找到保存的超参数文件: {best_params_path}")
            print("请先运行模式2进行调优，或使用模式1/2")
            return

        import json
        with open(best_params_path, 'r') as f:
            best_hyperparams = json.load(f)

        print("\n✅ 成功加载最佳超参数:")
        for key, value in best_hyperparams.items():
            print(f"  {key}: {value}")

    elif mode == '2':
        # ========== 超参数调优模式 ==========
        n_trials = int(input("输入调优次数 (建议30-100): ") or "50")

        study = run_hyperparameter_tuning(
            train_loader, val_loader, meta, cfg,
            n_trials=n_trials
        )

        best_hyperparams = study.best_params

        # ========== 新增：保存最佳超参数 ==========
        import json
        with open(best_params_path, 'w') as f:
            json.dump(best_hyperparams, f, indent=4)
        print(f"\n💾 最佳超参数已保存到: {best_params_path}")

    elif mode == '1':
        # ========== 默认模式 ==========
        print("\n使用默认超参数训练...")
        best_hyperparams = None

    else:
        print("无效选项，退出程序")
        return

    # ========== 统一训练流程 ==========
    print("\n" + "=" * 80)
    print("🎯 开始最终训练...")
    print("=" * 80)

    model, centers_np, history, _ = train_model_with_config(
        train_loader, val_loader, meta, cfg, best_hyperparams
    )

    # 3. Plot training history
    plot_training_history(history, cfg.RESULTS_DIR)

    # 4. Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)

    centers_torch = torch.FloatTensor(centers_np).to(cfg.DEVICE)

    # Get predictions
    pred_mu_scaled, pred_logvar_scaled, y_test_scaled = predict_on_loader(
        model, test_loader, centers_torch, cfg
    )

    # Transform to original space
    predictions_orig = transform_to_original_space(
        pred_mu_scaled, pred_logvar_scaled, meta['y_scaler']
    )

    # Get true values in original space
    y_test_logspace = meta['y_scaler'].inverse_transform(y_test_scaled)
    y_test_orig = np.exp(y_test_logspace)

    # 5. Compute metrics
    print("\n--- Metrics in Log-Space ---")
    metrics_log = compute_metrics(y_test_logspace, predictions_orig['mu_logspace'], cfg.TARGET_NAMES)
    print(metrics_log.to_string(index=False))
    metrics_log.to_csv(os.path.join(cfg.RESULTS_DIR, 'metrics_logspace.csv'), index=False)

    print("\n--- Metrics in Original Space ---")
    metrics_orig = compute_metrics(y_test_orig, predictions_orig['E_Y'], cfg.TARGET_NAMES)
    print(metrics_orig.to_string(index=False))
    metrics_orig.to_csv(os.path.join(cfg.RESULTS_DIR, 'metrics_original.csv'), index=False)

    # 6. Check monotonicity compliance
    print("\n--- Monotonicity Compliance Check ---")
    monotonicity_check = check_monotonicity(
        model, meta['X_test_scaled'], centers_torch,
        cfg.FEATURE_NAMES, cfg.MONOTONICITY_EXPECTATION,
        cfg, n_samples=200
    )
    print(monotonicity_check.to_string(index=False))
    monotonicity_check.to_csv(os.path.join(cfg.RESULTS_DIR, 'monotonicity_check.csv'), index=False)

    # 7. Generate diagnostic plots
    print("\n--- Generating Visualizations ---")

    # Residual diagnostics
    residual_diagnostics(
        y_test_logspace, predictions_orig['mu_logspace'],
        cfg.TARGET_NAMES, cfg.RESULTS_DIR
    )

    # Prediction plots
    plot_predictions(
        y_test_orig, predictions_orig,
        cfg.TARGET_NAMES, cfg.RESULTS_DIR
    )

    # Uncertainty distribution
    plot_uncertainty_distribution(
        predictions_orig, cfg.TARGET_NAMES, cfg.RESULTS_DIR
    )

    # Partial dependence plots
    plot_partial_dependence(
        model, centers_torch, meta['X_test_scaled'],
        meta['x_scaler'], cfg.FEATURE_NAMES, cfg.TARGET_NAMES,
        cfg, cfg.RESULTS_DIR, n_features_to_plot=6
    )

    # 8. Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"\nDataset sizes:")
    print(f"  Training:   {meta['n_train']} samples")
    print(f"  Validation: {meta['n_val']} samples")
    print(f"  Test:       {meta['n_test']} samples")

    print(f"\nModel configuration:")
    if best_hyperparams:
        print(f"  RBF centers:     {best_hyperparams['n_rbf']}")
        print(f"  Hidden dims:     [{best_hyperparams['hidden_dim1']}, {best_hyperparams['hidden_dim2']}]")
        print(f"  Physics penalty: {best_hyperparams['lambda_phys']:.6f}")
    else:
        print(f"  RBF centers:     {cfg.N_RBF}")
        print(f"  Hidden dims:     {cfg.HIDDEN_DIMS}")
        print(f"  Physics penalty: {cfg.LAMBDA_PHYS}")

    print(f"\nUncertainty statistics (Coefficient of Variation %):")
    for i, name in enumerate(cfg.TARGET_NAMES):
        cv = predictions_orig['Std_Y'][:, i] / (predictions_orig['E_Y'][:, i] + 1e-8) * 100
        print(f"  {name}: Mean = {cv.mean():.2f}%, Median = {np.median(cv):.2f}%")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"All results saved to: {cfg.RESULTS_DIR}/")
    print("=" * 60)

    # 9. Save predictions
    predictions_df = pd.DataFrame({
        f'True_{name}': y_test_orig[:, i]
        for i, name in enumerate(cfg.TARGET_NAMES)
    })
    for i, name in enumerate(cfg.TARGET_NAMES):
        predictions_df[f'Pred_Mean_{name}'] = predictions_orig['E_Y'][:, i]
        predictions_df[f'Pred_Std_{name}'] = predictions_orig['Std_Y'][:, i]
        predictions_df[f'Pred_CI_Lower_{name}'] = predictions_orig['CI_lower'][:, i]
        predictions_df[f'Pred_CI_Upper_{name}'] = predictions_orig['CI_upper'][:, i]

    predictions_df.to_csv(os.path.join(cfg.RESULTS_DIR, 'predictions.csv'), index=False)
    print(f"\nPredictions saved to: {cfg.RESULTS_DIR}/predictions.csv")

    # 10. 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'centers': centers_np,
        'hyperparameters': best_hyperparams,
        'x_scaler': meta['x_scaler'],
        'y_scaler': meta['y_scaler']
    }, cfg.MODEL_SAVE_PATH)
    print(f"✅ 最终模型已保存到: {cfg.MODEL_SAVE_PATH}")


# =============================================================================
#  SENSITIVITY ANALYSIS FOR λ_phy (惩罚权重敏感性分析)
# =============================================================================

def sensitivity_analysis_lambda_phy(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    meta: Dict,
    cfg: Config,
    lambda_values: List[float] = None,
    save_suffix: str = "lambda_sensitivity"
) -> None:
    """
    对物理惩罚权重 λ_phy 进行敏感性分析
    最小修改方案：复用现有 train_model_with_config 函数
    """
    if lambda_values is None:
        # 默认推荐的测试范围（可根据需要调整）
        lambda_values = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    print("\n" + "=" * 80)
    print("🔍 开始 λ_phy 惩罚权重敏感性分析")
    print("=" * 80)
    print(f"将测试以下 λ_phy 值: {lambda_values}")

    results_summary = []
    sensitivity_dir = os.path.join(cfg.RESULTS_DIR, save_suffix)
    os.makedirs(sensitivity_dir, exist_ok=True)

    # 使用当前最佳超参数（如果有的话）
    best_hyperparams = None
    best_params_path = os.path.join(cfg.RESULTS_DIR, 'best_hyperparameters.json')
    if os.path.exists(best_params_path):
        import json
        with open(best_params_path, 'r') as f:
            best_hyperparams = json.load(f)
        print("✅ 已加载最佳超参数作为基础配置")

    for i, lam in enumerate(lambda_values):
        print(f"\n[{i+1}/{len(lambda_values)}] 测试 λ_phy = {lam:.4f} ...")

        # 临时复制配置，避免污染全局 cfg
        import copy
        current_cfg = copy.deepcopy(cfg)
        current_cfg.LAMBDA_PHYS = lam

        # 训练模型（复用现有函数，几乎零修改）
        model, centers_np, history, best_val_nll = train_model_with_config(
            train_loader=train_loader,
            val_loader=val_loader,
            meta=meta,
            cfg=current_cfg,           # 传入修改后的 cfg
            hyperparams=best_hyperparams,
            verbose=False              # 关闭详细打印，避免输出过多
        )

        # 在测试集上评估
        centers_torch = torch.FloatTensor(centers_np).to(current_cfg.DEVICE)
        pred_mu_scaled, pred_logvar_scaled, y_test_scaled = predict_on_loader(
            model, test_loader, centers_torch, current_cfg
        )

        predictions_orig = transform_to_original_space(
            pred_mu_scaled, pred_logvar_scaled, meta['y_scaler']
        )
        y_test_logspace = meta['y_scaler'].inverse_transform(y_test_scaled)
        y_test_orig = np.exp(y_test_logspace)

        # 计算关键指标（使用标准化后的 log 空间便于对比）
        metrics_scaled = compute_metrics(
            y_test_scaled, pred_mu_scaled, current_cfg.TARGET_NAMES
        )

        # 单调性检查（这是 λ_phy 最直接影响的指标）
        mono_check = check_monotonicity(
            model, meta['X_test_scaled'], centers_torch,
            current_cfg.FEATURE_NAMES, current_cfg.MONOTONICITY_EXPECTATION,
            current_cfg, n_samples=200
        )
        avg_mono = mono_check['Compliance (%)'].mean()

        # 保存本次结果
        trial_dir = os.path.join(sensitivity_dir, f"lambda_{lam:.4f}")
        os.makedirs(trial_dir, exist_ok=True)

        # 保存历史（包含物理惩罚曲线，非常重要）
        joblib.dump(history, os.path.join(trial_dir, 'training_history.pkl'))
        plot_training_history(history, trial_dir)   # 会生成带物理惩罚的图

        # 保存指标
        metrics_scaled.to_csv(os.path.join(trial_dir, 'metrics.csv'), index=False)
        mono_check.to_csv(os.path.join(trial_dir, 'monotonicity.csv'), index=False)

        # 记录摘要
        results_summary.append({
            'lambda_phy': lam,
            'val_nll': best_val_nll,
            'avg_monotonicity (%)': avg_mono,
            'final_phys_penalty': history['train_phys'][-1],
            'test_r2_moment': metrics_scaled.iloc[0]['R²'],
            'test_r2_strain': metrics_scaled.iloc[1]['R²'] if len(metrics_scaled) > 1 else None
        })

        print(f"  → Val NLL: {best_val_nll:.6f} | Avg Monotonicity: {avg_mono:.1f}% | "
              f"Phys Penalty: {history['train_phys'][-1]:.6f}")

    # 生成整体敏感性总结表
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(os.path.join(sensitivity_dir, 'lambda_sensitivity_summary.csv'), index=False)

    print("\n" + "=" * 80)
    print("📊 λ_phy 敏感性分析完成！")
    print("=" * 80)
    print(summary_df.round(4).to_string(index=False))
    print(f"\n📁 所有结果已保存至: {sensitivity_dir}/")
    print("   - 每个 λ 值都有独立的训练历史图和指标")
    print("   - summary.csv 包含 NLL、单调性、R² 等关键对比")
# =============================================================================
# 12. UPDATED MAIN FUNCTION WITH COMPARISON
# =============================================================================

def main_with_comparison():
    """主执行流程（包含对比分析）"""
    print("\n" + "=" * 80)
    print("PHYSICS-INFORMED HETEROSCEDASTIC NEURAL NETWORK FOR MES PREDICTION")
    print("WITH COMPREHENSIVE BASELINE COMPARISON")
    print("=" * 80)

    # 检查数据文件
    if not os.path.exists(cfg.DATA_PATH):
        print(f"\nERROR: Data file not found at {cfg.DATA_PATH}")
        return

    # 1. 加载数据
    train_loader, val_loader, test_loader, meta = load_and_preprocess_data(
        cfg.DATA_PATH, cfg
    )

    # 2. 选择运行模式
    best_params_path = os.path.join(cfg.RESULTS_DIR, 'best_hyperparameters.json')

    print("\n请选择运行模式:")
    print("1. 使用默认超参数训练")
    print("2. 自动调优超参数")
    print("3. 加载已保存的最佳超参数并训练 (推荐)")
    mode = input("请输入选项 (1/2/3): ").strip()

    best_hyperparams = None

    if mode == '3':
        if not os.path.exists(best_params_path):
            print(f"\n❌ 未找到保存的超参数文件: {best_params_path}")
            print("请先运行模式2进行调优，或使用模式1")
            return

        import json
        with open(best_params_path, 'r') as f:
            best_hyperparams = json.load(f)

        print("\n✅ 成功加载最佳超参数:")
        for key, value in best_hyperparams.items():
            print(f"  {key}: {value}")

    elif mode == '2':
        n_trials = int(input("输入调优次数 (建议30-100): ") or "50")
        study = run_hyperparameter_tuning(
            train_loader, val_loader, meta, cfg, n_trials=n_trials
        )
        best_hyperparams = study.best_params

        import json
        with open(best_params_path, 'w') as f:
            json.dump(best_hyperparams, f, indent=4)
        print(f"\n💾 最佳超参数已保存到: {best_params_path}")

    elif mode == '1':
        print("\n使用默认超参数训练...")

    else:
        print("无效选项，退出程序")
        return

    # 3. 训练本文模型
    print("\n" + "=" * 80)
    print("🎯 开始训练本文模型...")
    print("=" * 80)

    model, centers_np, history, _ = train_model_with_config(
        train_loader, val_loader, meta, cfg, best_hyperparams
    )

    # 4. 绘制训练历史
    plot_training_history(history, cfg.RESULTS_DIR)

    # 5. **执行完整对比分析**
    print("\n" + "=" * 80)
    print("🔬 执行与基线模型的全面对比分析...")
    print("=" * 80)

    comparison_results = run_comprehensive_comparison(
        train_loader, val_loader, test_loader,
        model, centers_np, meta, cfg
    )

    # 6. 生成本文模型的详细诊断（原有功能）
    print("\n" + "=" * 80)
    print("📈 生成本文模型的详细诊断...")
    print("=" * 80)

    centers_torch = torch.FloatTensor(centers_np).to(cfg.DEVICE)


    # 预测
    pred_mu_scaled, pred_logvar_scaled, y_test_scaled = predict_on_loader(
        model, test_loader, centers_torch, cfg
    )

    predictions_orig = transform_to_original_space(
        pred_mu_scaled, pred_logvar_scaled, meta['y_scaler']
    )

    y_test_logspace = meta['y_scaler'].inverse_transform(y_test_scaled)
    y_test_orig = np.exp(y_test_logspace)

    # 残差诊断
    residual_diagnostics(
        y_test_logspace, predictions_orig['mu_logspace'],
        cfg.TARGET_NAMES, cfg.RESULTS_DIR
    )

    # 预测vs真值图（本文模型单独）
    plot_predictions(
        y_test_orig, predictions_orig,
        cfg.TARGET_NAMES, cfg.RESULTS_DIR
    )

    # 不确定性分布
    plot_uncertainty_distribution(
        predictions_orig, cfg.TARGET_NAMES, cfg.RESULTS_DIR
    )

    # 偏依赖图（本文模型单独）
    plot_partial_dependence(
        model, centers_torch, meta['X_test_scaled'],
        meta['x_scaler'], cfg.FEATURE_NAMES, cfg.TARGET_NAMES,
        cfg, cfg.RESULTS_DIR, n_features_to_plot=6
    )

    # 7. 保存最终结果
    print("\n" + "=" * 80)
    print("💾 保存结果...")
    print("=" * 80)

    # 保存预测结果
    predictions_df = pd.DataFrame({
        f'True_{name}': y_test_orig[:, i]
        for i, name in enumerate(cfg.TARGET_NAMES)
    })
    for i, name in enumerate(cfg.TARGET_NAMES):
        predictions_df[f'Pred_Mean_{name}'] = predictions_orig['E_Y'][:, i]
        predictions_df[f'Pred_Std_{name}'] = predictions_orig['Std_Y'][:, i]
        predictions_df[f'Pred_CI_Lower_{name}'] = predictions_orig['CI_lower'][:, i]
        predictions_df[f'Pred_CI_Upper_{name}'] = predictions_orig['CI_upper'][:, i]

    predictions_df.to_csv(os.path.join(cfg.RESULTS_DIR, 'predictions.csv'), index=False)

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'centers': centers_np,
        'hyperparameters': best_hyperparams,
        'x_scaler': meta['x_scaler'],
        'y_scaler': meta['y_scaler']
    }, cfg.MODEL_SAVE_PATH)

    print("\n" + "=" * 80)
    print("🎉 全部流程完成！")
    print("=" * 80)
    print(f"\n📁 所有结果已保存到: {cfg.RESULTS_DIR}/")
    print("\n生成的文件包括:")
    print("  1. 预测精度对比:")
    print("     - comparison_predictions.png (3列对比图)")
    print("     - metrics_comparison.csv (指标对比表)")
    print("  2. 不确定性量化:")
    print("     - comparison_uncertainty.png (带误差棒)")
    print("     - uncertainty_quantification_comparison.csv")
    print("  3. 物理一致性:")
    print("     - monotonicity_comparison.csv (单调性对比)")
    print("     - comparison_partial_dependence_*.png (偏依赖图)")
    print("  4. 本文模型详细诊断:")
    print("     - training_history.png")
    print("     - residual_diagnostics.png")
    print("     - predictions_vs_true.png")
    print("     - uncertainty_distribution.png")
    print("     - partial_dependence.png")
    print("  5. 模型和数据:")
    print("     - best_mes_model.pt (模型权重)")
    print("     - predictions.csv (详细预测)")
    export_dir = export_for_reliability_analysis(
        model=model,
        X_train=meta['X_train_scaled'],
        X_test=meta['X_test_scaled'],
        y_train=meta['y_train_scaled'],
        y_test=meta['y_test_scaled'],
        meta=meta,  # 确保包含centers
        save_dir=cfg.RESULTS_DIR
    )

    print(f"\n✅ Exported to: {export_dir}")
    print("   Next step: Update MES_OUTPUT_DIR in reliability_analysis_full.py")



# =============================================================================
# 14. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # 如果只想运行原始流程（不做对比）
    # main()

    # 如果想运行完整对比流程（推荐）
    # main_with_comparison()

    # 新增：敏感性分析（推荐放在完整训练之后）
    print("\n是否要立即运行 λ_phy 敏感性分析？ (y/n)")
    if input().strip().lower() == 'y':
        # 先运行一次完整训练以获得最佳超参数（推荐）
        train_loader, val_loader, test_loader, meta = load_and_preprocess_data(
            cfg.DATA_PATH, cfg
        )
        # ...（此处可复用你已有的训练代码得到 model 等，或直接调用下面）

        sensitivity_analysis_lambda_phy(
            train_loader, val_loader, test_loader, meta, cfg,
            lambda_values=[0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],  # 可自定义
            save_suffix="lambda_sensitivity"
        )



