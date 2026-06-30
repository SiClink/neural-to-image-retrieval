import csv
import os

import numpy as np

from utils.io import save_json, save_npz


SPLIT_INFO_NAME = "retrieval_split_info.json"
NORMALIZERS_NAME = "sessionwise_retrieval_normalizers.npz"
HISTORY_NAME = "history.json"
BEST_VAL_PREDICTIONS_NAME = "best_val_predictions.npz"
FINAL_TEST_PREDICTIONS_NAME = "final_test_retrieval_predictions.npz"
FINAL_TEST_ROWS_CSV_NAME = "final_test_centroid_prediction_rows.csv"
SUMMARY_NAME = "summary.json"

MODEL_NAME = "UnitTransformerRetrievalEncoder_RegionWiseSelfAttention_CentroidLoss"


def build_run_name(model_config, target_regions, train_ratio, val_ratio, test_ratio, seed):
    region_part = "_".join(target_regions)
    region_sa_tag = (
        f"regionSA{model_config.region_sa_layers}L"
        if model_config.use_region_self_attn
        else "noRegionSA"
    )
    return (
        f"unit_transformer_centroid_trialsplit_118_{region_part}_{region_sa_tag}_"
        f"ratio{train_ratio:.2f}_{val_ratio:.2f}_{test_ratio:.2f}_"
        f"seed{seed}"
    )


def write_split_info(
    save_dir,
    session_ids,
    session_files,
    feature_file,
    mapping_mode,
    target_regions,
    input_unit_stats,
    proj_dim,
    split_indices,
    train_labels,
    val_labels,
    test_labels,
    sessions,
    data_config,
    model_config,
):
    split_info = {
        "session_ids": session_ids,
        "session_files": session_files,
        "feature_file": feature_file,
        "feature_mapping_mode": mapping_mode,
        "target_regions": target_regions,
        "input_unit_stats": input_unit_stats,
        "proj_dim": int(proj_dim),
        "split_mode": "trial_split_per_label",
        "train_ratio": float(data_config.train_ratio),
        "val_ratio": float(data_config.val_ratio),
        "test_ratio": float(data_config.test_ratio),
        "train_labels": [int(x) for x in train_labels.tolist()],
        "val_labels": [int(x) for x in val_labels.tolist()],
        "test_labels": [int(x) for x in test_labels.tolist()],
        "no_class_overlap": False,
        "all_splits_share_same_labels": True,
        "train_trials": int(len(split_indices["train"])),
        "val_trials": int(len(split_indices["val"])),
        "test_trials": int(len(split_indices["test"])),
        "selected_unit_info": {sess["session_id"]: sess["selected_unit_info"] for sess in sessions},
        "model": MODEL_NAME,
        "task": "diagnostic image-level centroid retrieval; all 118 classes are shared by train/val/test and trials are split within each label",
        "input": "variable-length unit tokens: spike embedding + 3D CCF sin/cos position embedding + region embedding; region-wise unit self-attention inside each brain area; global Transformer encoder; masked pooling",
        "region_self_attention": {
            "enabled": bool(model_config.use_region_self_attn),
            "layers": int(model_config.region_sa_layers),
            "nhead": int(model_config.region_sa_nhead),
            "dim_feedforward": int(model_config.region_sa_dim_feedforward),
            "share_weights": bool(model_config.region_sa_share_weights),
        },
        "loss": "trial embedding -> label centroid; prototype CE + cosine/MSE alignment",
    }
    path = os.path.join(save_dir, SPLIT_INFO_NAME)
    save_json(path, split_info)
    return path, split_info


def write_normalizers_npz(
    save_dir,
    sessions,
    spike_normalizers,
    ccf_normalizers,
    train_labels,
    val_labels,
    test_labels,
    train_candidate_rows,
    val_candidate_rows,
    test_candidate_rows,
    proj_dim,
):
    npz_dict = {
        "train_labels": train_labels,
        "val_labels": val_labels,
        "test_labels": test_labels,
        "train_candidate_rows": train_candidate_rows,
        "val_candidate_rows": val_candidate_rows,
        "test_candidate_rows": test_candidate_rows,
        "proj_dim": np.array([proj_dim], dtype=np.int64),
    }

    for sess in sessions:
        sid = sess["session_id"]
        npz_dict[f"{sid}_spike_mean"] = spike_normalizers[sid]["spike_mean"]
        npz_dict[f"{sid}_spike_std"] = spike_normalizers[sid]["spike_std"]
        npz_dict[f"{sid}_ccf_mean"] = ccf_normalizers[sid]["ccf_mean"]
        npz_dict[f"{sid}_ccf_std"] = ccf_normalizers[sid]["ccf_std"]
        npz_dict[f"{sid}_selected_unit_indices"] = spike_normalizers[sid]["selected_unit_indices"]

    path = os.path.join(save_dir, NORMALIZERS_NAME)
    save_npz(path, **npz_dict)
    return path


def write_history_json(save_dir, history):
    path = os.path.join(save_dir, HISTORY_NAME)
    save_json(path, history)
    return path


def write_predictions_npz(save_dir, filename, metrics):
    path = os.path.join(save_dir, filename)
    save_npz(
        path,
        pred_raw_labels=metrics["pred_raw_labels"],
        true_raw_labels=metrics["true_raw_labels"],
        feature_rows=metrics["feature_rows"],
        centroid_features=metrics["centroid_features"],
        trial_raw_labels=metrics["trial_raw_labels"],
        session_idx=metrics["session_idx"],
        trial_idx=metrics["trial_idx"],
        spike_features=metrics["spike_features"],
    )
    return path


def write_prediction_rows_csv(save_dir, metrics):
    path = os.path.join(save_dir, FINAL_TEST_ROWS_CSV_NAME)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_raw_label", "pred_raw_label", "correct", "feature_row"])
        for true_y, pred_y, feat_row in zip(
            metrics["true_raw_labels"],
            metrics["pred_raw_labels"],
            metrics["feature_rows"],
        ):
            writer.writerow([int(true_y), int(pred_y), int(true_y == pred_y), int(feat_row)])
    return path


def write_summary_json(
    save_dir,
    cfg,
    mapping_mode,
    sessions,
    session_ids,
    target_regions,
    input_unit_stats,
    proj_dim,
    split_indices,
    train_labels,
    val_labels,
    test_labels,
    test_raw_labels,
    training_result,
):
    final_test_metrics = training_result["final_test_metrics"]
    best_path = training_result["best_path"]
    last_path = training_result["last_path"]
    final_predictions_path = training_result["final_test_predictions_path"]
    csv_path = training_result["final_test_prediction_rows_csv"]
    normalizer_path = os.path.join(save_dir, NORMALIZERS_NAME)

    summary = {
        "save_dir": save_dir,
        "model": MODEL_NAME,
        "task": "diagnostic image-level centroid retrieval",
        "split": "all 118 image classes are shared by train/val/test; trials are split within each label",
        "train_ratio": float(cfg.data.train_ratio),
        "val_ratio": float(cfg.data.val_ratio),
        "test_ratio": float(cfg.data.test_ratio),
        "input": "spike_count + CCF 3D sin/cos position embedding + region embedding, variable-length unit tokens, region-wise unit self-attention, global Transformer encoder, masked pooling",
        "region_self_attention": {
            "enabled": bool(cfg.model.use_region_self_attn),
            "layers": int(cfg.model.region_sa_layers),
            "nhead": int(cfg.model.region_sa_nhead),
            "dim_feedforward": int(cfg.model.region_sa_dim_feedforward),
            "share_weights": bool(cfg.model.region_sa_share_weights),
        },
        "feature_file": cfg.path.feature_file,
        "feature_mapping_mode": mapping_mode,
        "num_sessions": int(len(sessions)),
        "session_ids": session_ids,
        "target_regions": target_regions,
        "input_unit_stats": input_unit_stats,
        "proj_dim": int(proj_dim),
        "train_classes": int(len(train_labels)),
        "val_classes": int(len(val_labels)),
        "test_classes": int(len(test_labels)),
        "train_trials": int(len(split_indices["train"])),
        "val_trials": int(len(split_indices["val"])),
        "test_trials": int(len(split_indices["test"])),
        "best_epoch": int(training_result["best_epoch"]),
        "best_val_top1": float(training_result["best_val_top1"]),
        "test_chance_top1": float(1.0 / len(test_raw_labels)),
        "test_chance_top5": float(min(5, len(test_raw_labels)) / len(test_raw_labels)),
        "final_test_loss": float(final_test_metrics["loss"]),
        "final_test_top1": float(final_test_metrics["top1"]),
        "final_test_top5": float(final_test_metrics["top5"]),
        "final_test_mean_rank": float(final_test_metrics["mean_rank"]),
        "best_model": best_path,
        "last_model": last_path,
        "final_test_predictions_npz": final_predictions_path,
        "final_test_prediction_rows_csv": csv_path,
        "normalizer_npz": normalizer_path,
        "note": "Added region-wise unit self-attention before the global Transformer. Training/evaluation still use trial embedding -> label centroid -> retrieval loss/metrics. Best checkpoint is selected by validation centroid Top1; final test is evaluated once after training.",
    }

    path = os.path.join(save_dir, SUMMARY_NAME)
    save_json(path, summary)
    return path, summary
