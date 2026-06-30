import argparse
import json
import os

from config import (
    DataConfig,
    ExperimentConfig,
    LoaderConfig,
    LossConfig,
    ModelConfig,
    OptimConfig,
    PathConfig,
    ProjectConfig,
)


def parse_args():
    defaults = ProjectConfig()
    parser = argparse.ArgumentParser(
        description="Train a multi-session mouse neural spike retrieval model."
    )

    path_group = parser.add_argument_group("paths and data")
    path_group.add_argument("--data-dir", type=str, default=defaults.path.data_dir, help="Directory containing *_selected.npz session files.")
    path_group.add_argument("--feature-file", type=str, default=defaults.path.feature_file, help="PyTorch file containing image embeddings.")
    path_group.add_argument("--save-root", type=str, default=defaults.path.save_root, help="Directory where experiment runs are written.")
    path_group.add_argument("--session-ids", type=str, default=None, help="Comma-separated session IDs. Defaults to the configured session list.")
    path_group.add_argument("--target-regions", type=str, default=",".join(defaults.data.target_regions), help="Comma-separated visual regions to include.")
    path_group.add_argument("--train-ratio", type=float, default=defaults.data.train_ratio, help="Fraction of trials assigned to train within each label.")
    path_group.add_argument("--val-ratio", type=float, default=defaults.data.val_ratio, help="Fraction of trials assigned to val within each label.")
    path_group.add_argument("--test-ratio", type=float, default=defaults.data.test_ratio, help="Fraction of trials assigned to test within each label.")

    # Kept for backward compatibility with older class-split commands.
    path_group.add_argument("--train-class-count", type=int, default=defaults.data.train_class_count, help=argparse.SUPPRESS)
    path_group.add_argument("--val-class-count", type=int, default=defaults.data.val_class_count, help=argparse.SUPPRESS)
    path_group.add_argument("--test-class-count", type=int, default=defaults.data.test_class_count, help=argparse.SUPPRESS)
    path_group.add_argument("--split-mode", type=str, default=defaults.data.split_mode, choices=["sorted", "random"], help=argparse.SUPPRESS)

    loader_group = parser.add_argument_group("data loading")
    loader_group.add_argument("--batch-size", type=int, default=defaults.loader.batch_size, help="Only used by standard val/test DataLoaders; train uses frame-balanced batch parameters.")
    loader_group.add_argument("--frames-per-batch", type=int, default=defaults.loader.frames_per_batch, help="Number of image labels sampled per training batch.")
    loader_group.add_argument("--trials-per-frame", type=int, default=defaults.loader.trials_per_frame, help="Number of trials sampled per image label during training.")
    loader_group.add_argument("--steps-per-epoch", type=int, default=defaults.loader.steps_per_epoch, help="If <= 0, estimate automatically from train_set size.")
    loader_group.add_argument("--num-workers", type=int, default=defaults.loader.num_workers, help="Number of worker processes for DataLoader instances.")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--d-model", type=int, default=defaults.model.d_model)
    model_group.add_argument("--mlp-hidden", type=int, default=defaults.model.mlp_hidden)
    model_group.add_argument("--dropout", type=float, default=defaults.model.dropout)
    model_group.add_argument("--nhead", type=int, default=defaults.model.nhead)
    model_group.add_argument("--num-layers", type=int, default=defaults.model.num_layers)
    model_group.add_argument("--dim-feedforward", type=int, default=defaults.model.dim_feedforward)
    model_group.add_argument("--pooling", type=str, default=defaults.model.pooling, choices=["attn", "mean"])
    model_group.add_argument(
        "--use-region-self-attn",
        action="store_true",
        default=defaults.model.use_region_self_attn,
        help="Whether to enable within-region unit self-attention. Enabled by default.",
    )
    model_group.add_argument(
        "--no-region-self-attn",
        action="store_false",
        dest="use_region_self_attn",
        help="Disable within-region unit self-attention and fall back to the original global Transformer.",
    )
    model_group.add_argument("--region-sa-layers", type=int, default=defaults.model.region_sa_layers, help="Number of TransformerEncoder layers for within-region self-attention.")
    model_group.add_argument("--region-sa-nhead", type=int, default=defaults.model.region_sa_nhead, help="Number of heads for within-region self-attention; 0 reuses --nhead.")
    model_group.add_argument("--region-sa-dim-feedforward", type=int, default=defaults.model.region_sa_dim_feedforward, help="FFN hidden dimension for within-region self-attention.")
    model_group.add_argument(
        "--region-sa-share-weights",
        action="store_true",
        help="Share one within-region self-attention module across all brain regions; without this flag, each region has separate parameters.",
    )

    optim_group = parser.add_argument_group("optimization")
    optim_group.add_argument("--epochs", type=int, default=defaults.optim.epochs)
    optim_group.add_argument("--lr", type=float, default=defaults.optim.lr)
    optim_group.add_argument("--weight-decay", type=float, default=defaults.optim.weight_decay)
    optim_group.add_argument("--patience", type=int, default=defaults.optim.patience)
    optim_group.add_argument("--scheduler", type=str, default=defaults.optim.scheduler, choices=["cosine", "step"])
    optim_group.add_argument("--scheduler-step-size", type=int, default=defaults.optim.scheduler_step_size)
    optim_group.add_argument("--scheduler-gamma", type=float, default=defaults.optim.scheduler_gamma)
    optim_group.add_argument("--alpha", type=float, default=defaults.loss.alpha, help="Image/text loss weight. 1.0 uses only image features.")
    optim_group.add_argument("--ce-weight", type=float, default=defaults.loss.ce_weight)
    optim_group.add_argument("--cosine-weight", type=float, default=defaults.loss.cosine_weight)
    optim_group.add_argument("--mse-weight", type=float, default=defaults.loss.mse_weight)

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument("--seed", type=int, default=defaults.experiment.seed)
    runtime_group.add_argument("--cpu", action="store_true", help="Force CPU execution even when CUDA is available.")

    return parser.parse_args()

def build_config_from_args(args):
    defaults = ProjectConfig()
    target_regions = [x.strip() for x in args.target_regions.split(",") if x.strip() != ""]
    if len(target_regions) == 0:
        raise ValueError("target_regions cannot be empty")

    if args.region_sa_layers < 0:
        raise ValueError("--region-sa-layers cannot be less than 0")

    if args.region_sa_nhead <= 0:
        args.region_sa_nhead = args.nhead
    if args.region_sa_layers == 0:
        args.use_region_self_attn = False

    return ProjectConfig(
        path=PathConfig(
            data_dir=args.data_dir,
            feature_file=args.feature_file,
            save_root=args.save_root,
        ),
        data=DataConfig(
            default_session_ids=defaults.data.default_session_ids,
            target_regions=target_regions,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            train_class_count=args.train_class_count,
            val_class_count=args.val_class_count,
            test_class_count=args.test_class_count,
            split_mode=args.split_mode,
        ),
        loader=LoaderConfig(
            batch_size=args.batch_size,
            frames_per_batch=args.frames_per_batch,
            trials_per_frame=args.trials_per_frame,
            steps_per_epoch=args.steps_per_epoch,
            num_workers=args.num_workers,
        ),
        model=ModelConfig(
            d_model=args.d_model,
            mlp_hidden=args.mlp_hidden,
            dropout=args.dropout,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            pooling=args.pooling,
            use_region_self_attn=args.use_region_self_attn,
            region_sa_layers=args.region_sa_layers,
            region_sa_nhead=args.region_sa_nhead,
            region_sa_dim_feedforward=args.region_sa_dim_feedforward,
            region_sa_share_weights=args.region_sa_share_weights,
        ),
        optim=OptimConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            scheduler=args.scheduler,
            scheduler_step_size=args.scheduler_step_size,
            scheduler_gamma=args.scheduler_gamma,
        ),
        loss=LossConfig(
            alpha=args.alpha,
            ce_weight=args.ce_weight,
            cosine_weight=args.cosine_weight,
            mse_weight=args.mse_weight,
        ),
        experiment=ExperimentConfig(
            seed=args.seed,
            cpu=args.cpu,
        ),
    )


def get_device(use_cpu):
    import torch

    if use_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_dataloaders(sessions, split_indices, raw_label_to_feature_row, cfg, device):
    from torch.utils.data import DataLoader

    from data.dataset import FrameBalancedBatchSampler, MultiSessionRetrievalTokenDataset, collate_fn

    train_set = MultiSessionRetrievalTokenDataset(
        sessions,
        split_indices["train"],
        raw_label_to_feature_row,
    )
    val_set = MultiSessionRetrievalTokenDataset(
        sessions,
        split_indices["val"],
        raw_label_to_feature_row,
    )
    test_set = MultiSessionRetrievalTokenDataset(
        sessions,
        split_indices["test"],
        raw_label_to_feature_row,
    )

    train_batch_sampler = FrameBalancedBatchSampler(
        train_set,
        frames_per_batch=cfg.loader.frames_per_batch,
        trials_per_frame=cfg.loader.trials_per_frame,
        steps_per_epoch=cfg.loader.steps_per_epoch,
        seed=cfg.experiment.seed,
    )

    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_batch_sampler,
        num_workers=cfg.loader.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.loader.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.loader.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader, train_batch_sampler


def build_optimizer_and_scheduler(model, cfg):
    import torch

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    if cfg.optim.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.optim.epochs,
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.optim.scheduler_step_size,
            gamma=cfg.optim.scheduler_gamma,
        )
    return optimizer, scheduler


def main():
    args = parse_args()
    cfg = build_config_from_args(args)

    from data.loading import (
        build_indices_by_trial_split_per_label,
        build_raw_label_to_feature_row,
        build_target_unit_indices,
        get_session_files,
        load_feature_file,
        load_sessions,
        parse_session_ids,
    )
    from data.preprocessing import (
        fit_and_apply_sessionwise_ccf_normalizer,
        fit_and_apply_sessionwise_train_normalizer,
    )
    from models.unit_transformer import UnitTransformerRetrievalEncoder
    from training.centroid import build_candidate_tensors
    from training.engine import run_training
    from utils.experiment_writer import (
        build_run_name,
        write_normalizers_npz,
        write_split_info,
        write_summary_json,
    )
    from utils.io import ensure_dir, json_default
    from utils.seed import set_seed

    set_seed(cfg.experiment.seed)
    device = get_device(cfg.experiment.cpu)

    ensure_dir(cfg.path.save_root)
    img_features_all, text_features_all = load_feature_file(cfg.path.feature_file)
    if text_features_all is None:
        cfg.loss.alpha = 1.0
        args.alpha = 1.0

    session_ids = parse_session_ids(args.session_ids, cfg.data.default_session_ids)
    session_files = get_session_files(cfg.path.data_dir, session_ids)
    sessions = load_sessions(session_files, target_regions=cfg.data.target_regions)
    build_target_unit_indices(sessions, cfg.data.target_regions)

    ccf_normalizers = fit_and_apply_sessionwise_ccf_normalizer(sessions)

    split_indices, all_labels = build_indices_by_trial_split_per_label(
        sessions=sessions,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.experiment.seed,
    )
    train_labels = all_labels
    val_labels = all_labels
    test_labels = all_labels

    raw_label_to_feature_row, mapping_mode = build_raw_label_to_feature_row(
        all_labels=all_labels,
        img_features=img_features_all,
    )

    spike_normalizers, input_unit_stats = fit_and_apply_sessionwise_train_normalizer(
        sessions=sessions,
        split_indices=split_indices,
    )

    train_raw_labels, train_candidate_rows, train_raw_to_pos = build_candidate_tensors(
        train_labels,
        raw_label_to_feature_row,
    )
    val_raw_labels, val_candidate_rows, val_raw_to_pos = build_candidate_tensors(
        val_labels,
        raw_label_to_feature_row,
    )
    test_raw_labels, test_candidate_rows, test_raw_to_pos = build_candidate_tensors(
        test_labels,
        raw_label_to_feature_row,
    )
    candidates = {
        "train": {
            "raw_labels": train_raw_labels,
            "candidate_rows": train_candidate_rows,
            "raw_to_pos": train_raw_to_pos,
        },
        "val": {
            "raw_labels": val_raw_labels,
            "candidate_rows": val_candidate_rows,
            "raw_to_pos": val_raw_to_pos,
        },
        "test": {
            "raw_labels": test_raw_labels,
            "candidate_rows": test_candidate_rows,
            "raw_to_pos": test_raw_to_pos,
        },
    }

    proj_dim = int(img_features_all.shape[1])
    run_name = build_run_name(
        model_config=cfg.model,
        target_regions=cfg.data.target_regions,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.experiment.seed,
    )
    save_dir = ensure_dir(os.path.join(cfg.path.save_root, run_name))

    train_loader, val_loader, test_loader, train_batch_sampler = build_dataloaders(
        sessions,
        split_indices,
        raw_label_to_feature_row,
        cfg,
        device,
    )

    model = UnitTransformerRetrievalEncoder(
        proj_dim=proj_dim,
        d_model=cfg.model.d_model,
        mlp_hidden=cfg.model.mlp_hidden,
        num_regions=len(cfg.data.target_regions),
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        pooling=cfg.model.pooling,
        use_region_self_attn=cfg.model.use_region_self_attn,
        region_sa_layers=cfg.model.region_sa_layers,
        region_sa_nhead=cfg.model.region_sa_nhead,
        region_sa_dim_feedforward=cfg.model.region_sa_dim_feedforward,
        region_sa_share_weights=cfg.model.region_sa_share_weights,
    ).to(device)

    print("=" * 80)
    print(f"DEVICE: {device}")
    print(f"save_dir: {save_dir}")
    print(f"num_sessions: {len(sessions)}")
    print(f"target_regions: {cfg.data.target_regions}")
    print(f"input_unit_stats: {input_unit_stats}")
    print(f"proj_dim: {proj_dim}")
    print(
        "region_self_attention: "
        f"enabled={cfg.model.use_region_self_attn}, "
        f"layers={cfg.model.region_sa_layers}, "
        f"nhead={cfg.model.region_sa_nhead}, "
        f"dim_feedforward={cfg.model.region_sa_dim_feedforward}, "
        f"share_weights={cfg.model.region_sa_share_weights}"
    )
    print(f"train/val/test candidates: {len(train_raw_labels)}/{len(val_raw_labels)}/{len(test_raw_labels)}")
    print(f"val chance top1: {1.0 / len(val_raw_labels):.4f}, top5: {min(5, len(val_raw_labels)) / len(val_raw_labels):.4f}")
    print(f"test chance top1: {1.0 / len(test_raw_labels):.4f}, top5: {min(5, len(test_raw_labels)) / len(test_raw_labels):.4f}")
    print(model)
    print("=" * 80)

    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    write_split_info(
        save_dir=save_dir,
        session_ids=session_ids,
        session_files=session_files,
        feature_file=cfg.path.feature_file,
        mapping_mode=mapping_mode,
        target_regions=cfg.data.target_regions,
        input_unit_stats=input_unit_stats,
        proj_dim=proj_dim,
        split_indices=split_indices,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        sessions=sessions,
        data_config=cfg.data,
        model_config=cfg.model,
    )
    write_normalizers_npz(
        save_dir=save_dir,
        sessions=sessions,
        spike_normalizers=spike_normalizers,
        ccf_normalizers=ccf_normalizers,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        train_candidate_rows=train_candidate_rows,
        val_candidate_rows=val_candidate_rows,
        test_candidate_rows=test_candidate_rows,
        proj_dim=proj_dim,
    )

    training_result = run_training(
        model=model,
        train_loader=train_loader,
        train_batch_sampler=train_batch_sampler,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        img_features_all=img_features_all,
        text_features_all=text_features_all,
        candidates=candidates,
        cfg=cfg,
        save_dir=save_dir,
        input_unit_stats=input_unit_stats,
        proj_dim=proj_dim,
        session_ids=session_ids,
        target_regions=cfg.data.target_regions,
        raw_label_to_feature_row=raw_label_to_feature_row,
        args_dict=vars(args),
    )

    _, summary = write_summary_json(
        save_dir=save_dir,
        cfg=cfg,
        mapping_mode=mapping_mode,
        sessions=sessions,
        session_ids=session_ids,
        target_regions=cfg.data.target_regions,
        input_unit_stats=input_unit_stats,
        proj_dim=proj_dim,
        split_indices=split_indices,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        test_raw_labels=test_raw_labels,
        training_result=training_result,
    )

    print("\nSummary:")
    print(json.dumps(summary, indent=4, default=json_default))


if __name__ == "__main__":
    main()
