from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class PathConfig:
    """
    Path configuration.

    data_dir points to the directory containing session npz files,
    feature_file points to extracted_features.pt, and save_root is the parent
    directory for experiment run outputs.
    """
    data_dir: str = "data/selected_sessions"
    feature_file: str = "resources/extracted_features.pt"
    save_root: str = "runs"


@dataclass
class DataConfig:
    """
    Data split and target brain-region configuration.

    default_session_ids is the default list of training sessions. target_regions
    lists brain-region names present in the data, defaulting to
    ["VISp", "VISl", "VISrl"]. The current experiment uses a trial-level
    per-label split, so train/val/test all contain every label.
    train_class_count, val_class_count, test_class_count, and split_mode are
    kept only for backward CLI compatibility.
    """
    default_session_ids: list[str] = field(default_factory=lambda: [
        "715093703",
        "721123822",
        "743475441",
        "744228101",
        "750749662",
        "751348571",
        "756029989",
        "757216464",
        "763673393",
        "791319847",
        "798911424",
        "799864342",
    ])
    target_regions: list[str] = field(default_factory=lambda: ["VISp", "VISl", "VISrl"])
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    train_class_count: int = 72
    val_class_count: int = 18
    test_class_count: int = 28
    split_mode: str = "sorted"  # Options: "sorted" / "random"; legacy compatibility only for the current trial split.


@dataclass
class LoaderConfig:
    """
    DataLoader and training batch sampler configuration.

    The training set uses FrameBalancedBatchSampler, while validation and test
    sets use a standard DataLoader. Therefore, the training batch size is
    determined by frames_per_batch * trials_per_frame, and batch_size only
    affects val/test.
    """
    batch_size: int = 64  # Only used by val/test loaders; training batch = frames_per_batch * trials_per_frame.
    frames_per_batch: int = 72
    trials_per_frame: int = 1
    steps_per_epoch: int = 0  # 0 means estimating the number of steps per epoch from train_set size.
    num_workers: int = 4


@dataclass
class ModelConfig:
    """
    Model architecture configuration.

    Controls the unit-token Transformer, masked pooling, and optional
    region-wise unit self-attention. The pooling field can be "attn" or "mean".
    """
    d_model: int = 128
    mlp_hidden: int = 256
    dropout: float = 0.3
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    pooling: str = "attn"  # Options: "attn" / "mean".
    use_region_self_attn: bool = True
    region_sa_layers: int = 1  # 0 disables region self-attention.
    region_sa_nhead: int = 0  # 0 reuses the global Transformer's nhead.
    region_sa_dim_feedforward: int = 256
    region_sa_share_weights: bool = False


@dataclass
class OptimConfig:
    """
    Optimizer, training epoch, early stopping, and learning-rate schedule configuration.

    The optimizer is fixed to AdamW. The scheduler field can be "cosine" or
    "step".
    """
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-3
    patience: int = 20
    scheduler: str = "cosine"  # Options: "cosine" / "step".
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.5


@dataclass
class LossConfig:
    """
    Loss weights used for centroid retrieval training.

    Total loss = CE + cosine/MSE alignment. These fields only store weights;
    they do not change loss computation logic in the configuration layer.
    """
    alpha: float = 1.0  # 1.0 uses only image features; values below 1.0 mix in text features.
    ce_weight: float = 1.0
    cosine_weight: float = 0.2
    mse_weight: float = 0.05


@dataclass
class ExperimentConfig:
    """
    Experiment-level runtime configuration.

    seed controls randomness. cpu=True forces CPU execution; otherwise CUDA is
    selected automatically when available.
    """
    seed: int = 42
    cpu: bool = False


@dataclass
class ProjectConfig:
    """
    Top-level configuration object.

    Aggregates path, data, loader, model, optimizer, loss, and experiment
    configuration so main.py can pass them through the training pipeline.
    """
    path: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def config_to_dict(cfg: ProjectConfig) -> dict:
    return asdict(cfg)
