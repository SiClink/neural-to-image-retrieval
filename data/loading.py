import os

import numpy as np
import torch


def _as_float_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float()
    return torch.tensor(x, dtype=torch.float32)


def load_feature_file(feature_file):
    """
    Load image/text features from common extracted_features.pt layouts.

    image_features must be [num_images, feature_dim]. Text features are used
    only when they have the same shape as image_features.
    """
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"feature_file does not exist: {feature_file}")

    obj = torch.load(feature_file, map_location="cpu", weights_only=False)

    img_keys = [
        "image_features",
        "img_features",
        "img_features_all",
        "clip_image_features",
        "features",
    ]
    text_keys = [
        "text_features",
        "text_features_all",
        "clip_text_features",
    ]

    img_features = None
    text_features = None

    if isinstance(obj, torch.Tensor):
        img_features = obj.float()
    elif isinstance(obj, dict):
        for key in img_keys:
            if key in obj:
                img_features = _as_float_tensor(obj[key])
                break

        for key in text_keys:
            if key in obj:
                text_features = _as_float_tensor(obj[key])
                break

        if img_features is None:
            for value in obj.values():
                if isinstance(value, torch.Tensor) and value.ndim == 2:
                    img_features = value.detach().cpu().float()
                    break
    else:
        raise TypeError(f"Unsupported feature_file type: {type(obj)}")

    if img_features is None:
        raise KeyError("No image features found in feature_file. Please check the keys in extracted_features.pt.")
    if img_features.ndim != 2:
        raise ValueError(f"image_features must be [num_images, dim], got: {img_features.shape}")
    if text_features is not None and text_features.ndim != 2:
        raise ValueError(f"text_features must be [num_images, dim], got: {text_features.shape}")

    if text_features is not None and text_features.shape != img_features.shape:
        print(
            f"Warning: text_features shape {text_features.shape} and "
            f"img_features shape {img_features.shape} do not match; text_features will be ignored."
        )
        text_features = None

    print("image_features shape:", tuple(img_features.shape))
    if text_features is not None:
        print("text_features shape:", tuple(text_features.shape))
    else:
        print("text_features: None")

    return img_features, text_features


def build_raw_label_to_feature_row(all_labels, img_features):
    """
    Build raw label -> image feature row mapping.

    If raw labels can index feature rows directly, use them as rows. Otherwise,
    when feature rows equal class count, map sorted raw labels to row order.
    """
    all_labels = np.sort(np.asarray(all_labels, dtype=np.int64))
    max_label = int(all_labels.max())
    num_feature_rows = int(img_features.shape[0])

    raw_to_row = {}
    if max_label < num_feature_rows:
        for lab in all_labels:
            raw_to_row[int(lab)] = int(lab)
        mapping_mode = "raw_label_as_feature_row"
    elif num_feature_rows == len(all_labels):
        for i, lab in enumerate(all_labels):
            raw_to_row[int(lab)] = int(i)
        mapping_mode = "sorted_label_index_as_feature_row"
    else:
        raise ValueError(
            "Cannot automatically map labels to image feature rows: "
            f"max_label={max_label}, num_feature_rows={num_feature_rows}, num_labels={len(all_labels)}"
        )

    print("feature row mapping mode:", mapping_mode)
    return raw_to_row, mapping_mode


def parse_session_ids(session_ids_text, default_session_ids):
    if session_ids_text is None or session_ids_text.strip() == "":
        return list(default_session_ids)

    return [
        item.strip()
        for item in session_ids_text.split(",")
        if item.strip() != ""
    ]


def get_session_files(data_dir, session_ids):
    files = []
    for sid in session_ids:
        sid = str(sid)
        path = os.path.join(data_dir, f"{sid}_selected.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Session file not found: {path}")
        files.append(path)
    return files


def decode_region_array(region):
    return np.array([
        r.decode("utf-8") if isinstance(r, bytes) else str(r)
        for r in region
    ])


def load_sessions(session_files, target_regions=None):
    sessions = []

    for path in session_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Session file does not exist: {path}")

        data = np.load(path, allow_pickle=True)
        required_keys = ["spike_count", "labels", "region", "ccf"]
        for key in required_keys:
            if key not in data.files:
                raise KeyError(f"{path} is missing required field: {key}")

        spike_count = data["spike_count"].astype(np.float32)  # [num_trials, num_units]
        labels = data["labels"].astype(np.int64)              # [num_trials]
        region = decode_region_array(data["region"])          # [num_units]
        ccf = data["ccf"].astype(np.float32)                  # [num_units, 3]

        if spike_count.ndim != 2:
            raise ValueError(f"{path} spike_count must be [num_trials, num_units], got: {spike_count.shape}")
        if labels.ndim != 1:
            raise ValueError(f"{path} labels must be [num_trials], got: {labels.shape}")
        if ccf.ndim != 2 or ccf.shape[1] != 3:
            raise ValueError(f"{path} ccf must be [num_units, 3], got: {ccf.shape}")

        if spike_count.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{path} trial count and label count do not match: "
                f"spike_count={spike_count.shape}, labels={labels.shape}"
            )
        if spike_count.shape[1] != len(region):
            raise ValueError(
                f"{path} unit count and region count do not match: "
                f"spike_count={spike_count.shape}, region={len(region)}"
            )
        if spike_count.shape[1] != ccf.shape[0]:
            raise ValueError(
                f"{path} unit count and ccf row count do not match: "
                f"spike_count={spike_count.shape}, ccf={ccf.shape}"
            )

        session_id = os.path.basename(path).replace("_selected.npz", "")
        sessions.append({
            "session_id": session_id,
            "path": path,
            "spike_count_raw": spike_count,
            "labels": labels,
            "region": region,
            "ccf_raw": ccf,
            "num_trials": int(spike_count.shape[0]),
            "num_units": int(spike_count.shape[1]),
        })

        if target_regions is None:
            region_counts = {}
        else:
            region_counts = {r: int(np.sum(region == r)) for r in target_regions}

        print(
            f"Loaded session {session_id}: "
            f"spike_count={spike_count.shape}, "
            f"ccf={ccf.shape}, "
            f"num_classes={len(np.unique(labels))}, "
            f"region_counts={region_counts}"
        )

    return sessions


def build_target_unit_indices(sessions, target_regions):
    region_to_id = {r: i for i, r in enumerate(target_regions)}

    for sess in sessions:
        selected_indices = []
        selected_region_ids = []
        selected_unit_info = {}

        for region_name in target_regions:
            idx = np.where(sess["region"] == region_name)[0].astype(np.int64)

            # Stable order inside each region: finite CCF first, lexicographic AP/DV/LR.
            if len(idx) > 0:
                ccf_r = sess["ccf_raw"][idx]
                finite_mask = np.isfinite(ccf_r).all(axis=1)
                finite_idx = idx[finite_mask]
                nonfinite_idx = idx[~finite_mask]
                if len(finite_idx) > 0:
                    ccf_f = sess["ccf_raw"][finite_idx]
                    order = np.lexsort((ccf_f[:, 2], ccf_f[:, 1], ccf_f[:, 0]))
                    finite_idx = finite_idx[order]
                idx = np.concatenate([finite_idx, nonfinite_idx], axis=0).astype(np.int64)

            if len(idx) == 0:
                raise ValueError(f"session {sess['session_id']} has no units in target region {region_name}")

            selected_indices.append(idx)
            selected_region_ids.extend([region_to_id[region_name]] * len(idx))
            selected_unit_info[region_name] = {
                "selected": int(len(idx)),
                "indices": idx.astype(int).tolist(),
            }

        selected_indices = np.concatenate(selected_indices, axis=0).astype(np.int64)
        selected_region_ids = np.asarray(selected_region_ids, dtype=np.int64)

        sess["selected_unit_indices"] = selected_indices
        sess["selected_region_ids"] = selected_region_ids
        sess["selected_unit_info"] = selected_unit_info
        sess["selected_input_units"] = int(len(selected_indices))

        print(
            f"session {sess['session_id']} selected target units: "
            f"{len(selected_indices)} / {sess['num_units']}"
        )


def get_input_unit_stats(sessions):
    counts = [int(sess["selected_input_units"]) for sess in sessions]
    return {
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
        "mean": float(np.mean(counts)),
        "per_session": {sess["session_id"]: int(sess["selected_input_units"]) for sess in sessions},
    }


def build_indices_by_class_split(sessions, train_labels, val_labels, test_labels):
    train_label_set = set(int(x) for x in train_labels)
    val_label_set = set(int(x) for x in val_labels)
    test_label_set = set(int(x) for x in test_labels)

    split_indices = {"train": [], "val": [], "test": []}

    for sess_idx, sess in enumerate(sessions):
        labels = sess["labels"]
        for trial_idx, lab in enumerate(labels):
            lab = int(lab)
            if lab in train_label_set:
                split_indices["train"].append((sess_idx, trial_idx))
            elif lab in val_label_set:
                split_indices["val"].append((sess_idx, trial_idx))
            elif lab in test_label_set:
                split_indices["test"].append((sess_idx, trial_idx))

    for split in ["train", "val", "test"]:
        print(f"{split} trials: {len(split_indices[split])}")

    return split_indices


def build_indices_by_trial_split_per_label(
    sessions,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """
    Diagnostic split: every raw label appears in train/val/test; trials are split per label.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1, got: {total_ratio}")

    rng = np.random.default_rng(seed)

    label_to_indices = {}
    for sess_idx, sess in enumerate(sessions):
        labels = sess["labels"]
        for trial_idx, lab in enumerate(labels):
            lab = int(lab)
            label_to_indices.setdefault(lab, []).append((sess_idx, trial_idx))

    split_indices = {"train": [], "val": [], "test": []}

    for lab in sorted(label_to_indices.keys()):
        indices = list(label_to_indices[lab])
        order = rng.permutation(len(indices))
        indices = [indices[int(i)] for i in order]

        n = len(indices)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        else:
            n_train = max(1, n_train)
            n_val = max(0, min(n_val, n - n_train))

        train_part = indices[:n_train]
        val_part = indices[n_train:n_train + n_val]
        test_part = indices[n_train + n_val:]

        split_indices["train"].extend(train_part)
        split_indices["val"].extend(val_part)
        split_indices["test"].extend(test_part)

    all_labels = np.asarray(sorted(label_to_indices.keys()), dtype=np.int64)

    print("=" * 80)
    print("Trial-level diagnostic split complete:")
    print(f"total labels: {len(all_labels)}")
    print("train/val/test all contain every label; only trials are split, not image classes.")
    print(f"train_ratio={train_ratio}, val_ratio={val_ratio}, test_ratio={test_ratio}")
    print(f"all labels: {all_labels.tolist()}")
    print("=" * 80)

    for split in ["train", "val", "test"]:
        labels_in_split = []
        for sess_idx, trial_idx in split_indices[split]:
            labels_in_split.append(int(sessions[sess_idx]["labels"][trial_idx]))
        print(
            f"{split} trials: {len(split_indices[split])}, "
            f"labels: {len(np.unique(labels_in_split))}"
        )

    return split_indices, all_labels
