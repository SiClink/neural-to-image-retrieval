import numpy as np

from data.loading import get_input_unit_stats


def fit_and_apply_sessionwise_ccf_normalizer(sessions):
    """
    Normalize fixed unit CCF coordinates per session.

    Uses only selected target units. Invalid CCF rows are filled by the valid
    selected-unit mean, or zeros when all selected rows are invalid.
    """
    ccf_normalizers = {}

    for sess in sessions:
        selected_idx = sess["selected_unit_indices"]
        ccf_selected = sess["ccf_raw"][selected_idx].astype(np.float32)

        finite = np.isfinite(ccf_selected).all(axis=1)
        if not np.all(finite):
            if finite.any():
                valid_mean = ccf_selected[finite].mean(axis=0, keepdims=True)
            else:
                valid_mean = np.zeros((1, 3), dtype=np.float32)
            ccf_selected[~finite] = valid_mean

        ccf_mean = ccf_selected.mean(axis=0, keepdims=True).astype(np.float32)
        ccf_std = ccf_selected.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
        ccf_selected_norm = ((ccf_selected - ccf_mean) / ccf_std).astype(np.float32)

        sess["ccf_selected_norm"] = ccf_selected_norm
        ccf_normalizers[sess["session_id"]] = {
            "ccf_mean": ccf_mean,
            "ccf_std": ccf_std,
            "num_units_selected": int(len(selected_idx)),
        }

        print(
            f"CCF normalizer session {sess['session_id']}: "
            f"selected_units={len(selected_idx)}, "
            f"ccf_mean={ccf_mean.squeeze().tolist()}, "
            f"ccf_std={ccf_std.squeeze().tolist()}"
        )

    return ccf_normalizers


def fit_and_apply_sessionwise_train_normalizer(sessions, split_indices):
    """
    Spike preprocessing per session:
        log1p(max(x, 0)) -> per-unit z-score fitted on train trials only.

    Writes selected normalized spikes and selected region ids back into each session.
    """
    train_by_session = {i: [] for i in range(len(sessions))}
    for sess_idx, trial_idx in split_indices["train"]:
        train_by_session[sess_idx].append(trial_idx)

    normalizers = {}

    for sess_idx, sess in enumerate(sessions):
        spike = sess["spike_count_raw"]
        spike_log = np.log1p(np.maximum(spike, 0.0)).astype(np.float32)
        train_trials = np.asarray(train_by_session[sess_idx], dtype=np.int64)

        if len(train_trials) == 0:
            raise ValueError(
                f"session {sess['session_id']} has no training trials, so spike normalization parameters cannot be fitted."
            )

        spike_mean = spike_log[train_trials].mean(axis=0, keepdims=True).astype(np.float32)
        spike_std = spike_log[train_trials].std(axis=0, keepdims=True).astype(np.float32) + 1e-6
        spike_norm = ((spike_log - spike_mean) / spike_std).astype(np.float32)

        selected_idx = sess["selected_unit_indices"]
        spike_selected = spike_norm[:, selected_idx].astype(np.float32)

        sess["spike_count_selected_norm"] = spike_selected
        sess["region_ids_selected"] = sess["selected_region_ids"]

        normalizers[sess["session_id"]] = {
            "spike_mean": spike_mean,
            "spike_std": spike_std,
            "selected_unit_indices": selected_idx,
            "selected_unit_info": sess["selected_unit_info"],
            "num_units_raw": int(sess["num_units"]),
            "num_units_selected": int(len(selected_idx)),
        }

        print(
            f"Spike normalizer session {sess['session_id']}: "
            f"train_trials={len(train_trials)}, "
            f"raw_units={sess['num_units']}, "
            f"selected_units={len(selected_idx)}"
        )

    input_unit_stats = get_input_unit_stats(sessions)
    print("Variable-length unit count stats:", input_unit_stats)
    return normalizers, input_unit_stats
