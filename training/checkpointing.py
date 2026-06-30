import os
from dataclasses import asdict, is_dataclass

import torch


BEST_CHECKPOINT_NAME = "best_val_top1_retrieval_model.pt"
LAST_CHECKPOINT_NAME = "last_model.pt"


def _maybe_asdict(obj):
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def build_checkpoint_payload(
    model,
    model_config,
    session_ids,
    target_regions,
    train_labels,
    val_labels,
    test_labels,
    train_candidate_rows,
    val_candidate_rows,
    test_candidate_rows,
    raw_label_to_feature_row,
    input_unit_stats,
    proj_dim,
    args_dict,
    config_dict=None,
    best_val_top1=None,
    best_epoch=None,
):
    model_cfg = _maybe_asdict(model_config)

    payload = {
        "model_state_dict": model.state_dict(),
        "input_unit_stats": input_unit_stats,
        "proj_dim": int(proj_dim),
        "d_model": int(model_cfg["d_model"]),
        "mlp_hidden": int(model_cfg["mlp_hidden"]),
        "dropout": float(model_cfg["dropout"]),
        "nhead": int(model_cfg["nhead"]),
        "num_layers": int(model_cfg["num_layers"]),
        "dim_feedforward": int(model_cfg["dim_feedforward"]),
        "pooling": model_cfg["pooling"],
        "use_region_self_attn": bool(model_cfg["use_region_self_attn"]),
        "region_sa_layers": int(model_cfg["region_sa_layers"]),
        "region_sa_nhead": int(model_cfg["region_sa_nhead"]),
        "region_sa_dim_feedforward": int(model_cfg["region_sa_dim_feedforward"]),
        "region_sa_share_weights": bool(model_cfg["region_sa_share_weights"]),
        "model_config": model_cfg,
        "session_ids": session_ids,
        "target_regions": target_regions,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "test_labels": test_labels,
        "train_candidate_rows": train_candidate_rows,
        "val_candidate_rows": val_candidate_rows,
        "test_candidate_rows": test_candidate_rows,
        "raw_label_to_feature_row": raw_label_to_feature_row,
        "args": args_dict,
    }

    if config_dict is not None:
        payload["config"] = config_dict
    if best_val_top1 is not None:
        payload["best_val_top1"] = float(best_val_top1)
    if best_epoch is not None:
        payload["best_epoch"] = int(best_epoch)

    return payload


def save_checkpoint(payload, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path, map_location):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)
