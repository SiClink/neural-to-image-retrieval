import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================
# 1. Path settings
# ==========================
cache_dir = os.environ.get("ALLEN_ECEPHYS_CACHE_DIR", "data/ecephys_cache_dir")
output_root = os.environ.get("OUTPUT_ROOT", "data")
units_csv_path = os.path.join(cache_dir, "units.csv")
channels_csv_path = os.path.join(cache_dir, "channels.csv")

output_dir = os.path.join(output_root, "selected_sessions")
os.makedirs(output_dir, exist_ok=True)

# ==========================
# 2. Selection criteria
# ==========================
target_regions = ["VISp", "VISl", "VISrl"]
min_total_units = 213
min_units_per_region = 44

# ==========================
# 3. Read units.csv and channels.csv lookup tables
# ==========================
units_table = pd.read_csv(units_csv_path)
channels_table = pd.read_csv(channels_csv_path)

# channel_id -> ccf + region
channel_to_ccf = {}
for _, row in channels_table.iterrows():
    channel_id = int(row["id"])

    ccf = [
        float(row["anterior_posterior_ccf_coordinate"]),
        float(row["dorsal_ventral_ccf_coordinate"]),
        float(row["left_right_ccf_coordinate"])
    ]

    region = row["ecephys_structure_acronym"]

    channel_to_ccf[channel_id] = {
        "ccf": ccf,
        "region": region
    }

# unit_id -> ccf + region
unit_to_ccf = {}
for _, row in units_table.iterrows():
    unit_id = int(row["id"])
    channel_id = int(row["ecephys_channel_id"])

    if channel_id not in channel_to_ccf:
        raise ValueError(
            f"Unit {unit_id} references channel_id {channel_id}, which is not present in channels.csv."
        )

    unit_to_ccf[unit_id] = channel_to_ccf[channel_id]


# ==========================
# 4. Define single-session processing
# ==========================
def process_one_session(session_dir):
    session_id = os.path.basename(session_dir).replace("session_", "")
    nwb_path = os.path.join(session_dir, f"session_{session_id}.nwb")

    if not os.path.exists(nwb_path):
        return {
            "status": "NWB_MISSING",
            "reason": "NWB file does not exist",
            "session_id": session_id
        }

    try:
        with h5py.File(nwb_path, "r") as f:
            print(f"\nProcessing session {session_id}")

            # ==========================
            # 4.1 Extract Natural Scenes timing and labels
            # ==========================
            ns_group = f["intervals"]["natural_scenes_presentations"]

            starts = ns_group["start_time"][()]
            stops = ns_group["stop_time"][()]
            frames = ns_group["frame"][()]

            if "id" in ns_group:
                presentation_ids = ns_group["id"][()]
            else:
                presentation_ids = np.arange(len(frames))

            # Filter invalid stimuli where frame == -1
            valid_mask = frames != -1.0

            valid_starts = starts[valid_mask]
            valid_stops = stops[valid_mask]
            valid_frames = frames[valid_mask]
            valid_ids = presentation_ids[valid_mask]

            print(f"Valid image presentations: {len(valid_ids)}")

            # ==========================
            # 4.2 Build target labels y
            # ==========================
            targets = pd.Series(
                data=valid_frames,
                index=valid_ids,
                name="frame"
            )
            targets.index.name = "stimulus_presentation_id"

            labels = targets.values.astype(np.int64)
            stimulus_presentation_ids = targets.index.values.astype(np.int64)

            print(f"Label shape: {labels.shape}")
            print(f"Number of image classes: {len(np.unique(labels))}")

            # ==========================
            # 4.3 Read unit metadata
            # ==========================
            unit_ids_all = f["units"]["id"][()]
            peak_channel_ids = f["units"]["peak_channel_id"][()]
            unit_qualities = f["units"]["quality"][()]
            firing_rates = f["units"]["firing_rate"][()]

            # ==========================
            # 4.4 Select neurons from target brain regions
            # ==========================
            selected_unit_indices = []

            for idx, chan in enumerate(peak_channel_ids):
                quality = unit_qualities[idx]

                if isinstance(quality, bytes):
                    is_good = quality == b"good"
                else:
                    is_good = str(quality) == "good"

                if is_good and firing_rates[idx] > 0.5:
                    unit_id = int(unit_ids_all[idx])

                    if unit_id in unit_to_ccf:
                        region = unit_to_ccf[unit_id]["region"]

                        if region in target_regions:
                            selected_unit_indices.append(idx)

            # ==========================
            # 4.5 Count neurons by brain region
            # ==========================
            region_names = [
                unit_to_ccf[int(unit_ids_all[idx])]["region"]
                for idx in selected_unit_indices
            ]

            region_counts = pd.Series(region_names).value_counts().to_dict()

            for r in target_regions:
                region_counts.setdefault(r, 0)

            total_units = sum(region_counts.values())

            failed_reasons = []

            if total_units < min_total_units:
                failed_reasons.append(
                    f"Insufficient total neuron count: {total_units} < {min_total_units}"
                )

            for r in target_regions:
                if region_counts[r] < min_units_per_region:
                    failed_reasons.append(
                        f"Insufficient {r} neuron count: {region_counts[r]} < {min_units_per_region}"
                    )

            if failed_reasons:
                return {
                    "status": "FAILED",
                    "reason": "; ".join(failed_reasons),
                    "session_id": session_id
                }

            # ==========================
            # 4.6 Extract spike counts
            # ==========================
            spike_times_index = f["units"]["spike_times_index"][()]
            all_spike_times = f["units"]["spike_times"][()]

            selected_unit_ids = [
                int(unit_ids_all[idx])
                for idx in selected_unit_indices
            ]

            num_units = len(selected_unit_ids)
            num_presentations = len(valid_starts)

            spike_count = np.zeros(
                (num_presentations, num_units),
                dtype=np.float32
            )

            ccf_coords = []
            final_regions = []

            print(f"Session {session_id}: counting spikes within each stimulus time window...")

            for col_idx, row_idx in enumerate(selected_unit_indices):
                unit_id = int(unit_ids_all[row_idx])

                start_idx = 0 if row_idx == 0 else spike_times_index[row_idx - 1]
                end_idx = spike_times_index[row_idx]

                spikes = all_spike_times[start_idx:end_idx]

                start_indices = np.searchsorted(spikes, valid_starts)
                stop_indices = np.searchsorted(spikes, valid_stops)

                counts = stop_indices - start_indices

                if np.any(counts < 0):
                    raise ValueError(
                        f"Session {session_id}, unit {unit_id} produced a negative spike count"
                    )

                spike_count[:, col_idx] = counts

                ccf_coords.append(unit_to_ccf[unit_id]["ccf"])
                final_regions.append(unit_to_ccf[unit_id]["region"])

            # ==========================
            # 4.7 Check that X and y are aligned
            # ==========================
            if spike_count.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"X and y counts do not match: spike_count={spike_count.shape}, labels={labels.shape}"
                )

            return {
                "status": "OK",
                "session_id": session_id,

                # Features and labels
                "spike_count": spike_count,
                "labels": labels,
                "stimulus_presentation_ids": stimulus_presentation_ids,

                # Stimulus timing
                "valid_starts": valid_starts.astype(np.float64),
                "valid_stops": valid_stops.astype(np.float64),
                "valid_frames": valid_frames.astype(np.int64),

                # Neuron metadata
                "unit_ids": np.array(selected_unit_ids, dtype=np.int64),
                "ccf": np.array(ccf_coords, dtype=np.float32),
                "region": np.array(final_regions),

                # Summary statistics
                "region_counts": region_counts
            }

    except Exception as e:
        return {
            "status": "ERROR",
            "reason": str(e),
            "session_id": session_id
        }


# ==========================
# 5. Batch processing
# ==========================
session_dirs = [
    os.path.join(cache_dir, d)
    for d in os.listdir(cache_dir)
    if d.startswith("session_")
]
session_dirs.sort()

selected_sessions = []
failed_sessions = []

for session_dir in tqdm(session_dirs, desc="Processing sessions"):
    result = process_one_session(session_dir)
    session_id = result["session_id"]

    if result["status"] == "OK":
        save_path = os.path.join(output_dir, f"{session_id}_selected.npz")

        np.savez(
            save_path,

            # X and y
            spike_count=result["spike_count"],
            labels=result["labels"],

            # Stimulus metadata
            stimulus_presentation_ids=result["stimulus_presentation_ids"],
            valid_starts=result["valid_starts"],
            valid_stops=result["valid_stops"],
            valid_frames=result["valid_frames"],

            # Neuron metadata
            unit_ids=result["unit_ids"],
            ccf=result["ccf"],
            region=result["region"]
        )

        selected_sessions.append(session_id)

        print(
            f"Saved session {session_id}, "
            f"X: {result['spike_count'].shape}, "
            f"y: {result['labels'].shape}, "
            f"image_classes: {len(np.unique(result['labels']))}, "
            f"total_neurons: {sum(result['region_counts'].values())}, "
            f"regions: {result['region_counts']}"
        )

    else:
        failed_sessions.append({
            "session_id": session_id,
            "status": result["status"],
            "reason": result["reason"]
        })

        print(
            f"session {session_id} did not meet criteria or failed to read: "
            f"{result['status']} | {result['reason']}"
        )


# ==========================
# 6. Save log files
# ==========================
log_path_selected = os.path.join(output_dir, "selected_sessions.txt")
with open(log_path_selected, "w") as f:
    for sid in selected_sessions:
        f.write(sid + "\n")

log_path_failed = os.path.join(output_dir, "failed_sessions.txt")
with open(log_path_failed, "w") as f:
    for s in failed_sessions:
        f.write(
            f"{s['session_id']}\t{s['status']}\t{s['reason']}\n"
        )

print(f"\nSelected {len(selected_sessions)} sessions in total")
print(f"Selected-session log saved to: {log_path_selected}")

print(f"{len(failed_sessions)} sessions did not meet criteria or failed to read")
print(f"Failed-session log saved to: {log_path_failed}")
