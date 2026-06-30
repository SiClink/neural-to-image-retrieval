import os
import json
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


# =========================
# 1. Path settings
# =========================
cache_dir = os.environ.get("ALLEN_ECEPHYS_CACHE_DIR", "data/ecephys_cache_dir")
output_json = os.path.join(cache_dir, "session_matrix_summary.json")

target_regions = ["VISp", "VISl", "VISrl"]
region_order = ["VISp", "VISl", "VISrl"]

# =========================
# Read sessions.csv
# =========================
sessions_csv_path = os.path.join(cache_dir, "sessions.csv")

if os.path.exists(sessions_csv_path):
    sessions_df = pd.read_csv(sessions_csv_path)

    # Some CSV exports use id, while others use ecephys_session_id.
    if "ecephys_session_id" in sessions_df.columns:
        session_id_col = "ecephys_session_id"
    elif "id" in sessions_df.columns:
        session_id_col = "id"
    else:
        raise ValueError(f"sessions.csv does not contain a session id column; columns: {sessions_df.columns.tolist()}")

    sessions_df[session_id_col] = sessions_df[session_id_col].astype(str)

    session_type_map = dict(
        zip(
            sessions_df[session_id_col],
            sessions_df["session_type"]
        )
    )

    print(f"Loaded sessions.csv with {len(sessions_df)} sessions")
else:
    session_type_map = {}
    print(f"sessions.csv not found: {sessions_csv_path}")

def process_one_session(session_dir, target_regions):
    """
    Read one session_xxx directory and summarize natural-scenes matrix sizes.
    """
    session_name = os.path.basename(session_dir)

    if not session_name.startswith("session_"):
        return None

    session_id = session_name.replace("session_", "")
    nwb_path = os.path.join(session_dir, f"session_{session_id}.nwb")

    result = {
        "session_id": session_id,
        "session_type": session_type_map.get(str(session_id), "unknown"),
        "session_dir": session_dir,
        "nwb_path": nwb_path,
        "status": "failed",
        "error": None,
        "num_presentations": None,
        "num_units": None,
        "design_matrix_shape": None,
        "targets_shape": None,
        "num_image_classes": None,
        "region_unit_counts": {},
    }

    if not os.path.exists(nwb_path):
        result["error"] = "NWB file not found"
        return result

    try:
        with h5py.File(nwb_path, "r") as f:
            # =========================
            # 2. Check whether natural_scenes exists
            # =========================
            if "intervals" not in f:
                result["error"] = "No intervals group in NWB"
                return result

            if "natural_scenes_presentations" not in f["intervals"]:
                result["error"] = "No natural_scenes_presentations in this session"
                return result

            ns_group = f["intervals"]["natural_scenes_presentations"]

            starts = ns_group["start_time"][()]
            stops = ns_group["stop_time"][()]
            frames = ns_group["frame"][()]

            if "id" in ns_group:
                presentation_ids = ns_group["id"][()]
            else:
                presentation_ids = np.arange(len(frames))

            # Filter blank stimuli where frame == -1
            valid_mask = frames != -1.0
            valid_starts = starts[valid_mask]
            valid_stops = stops[valid_mask]
            valid_frames = frames[valid_mask]
            valid_ids = presentation_ids[valid_mask]

            targets = pd.Series(
                data=valid_frames,
                index=valid_ids,
                name="frame"
            )
            targets.index.name = "stimulus_presentation_id"

            # =========================
            # 3. Read unit and electrode metadata
            # =========================
            required_paths = [
                ("units", "id"),
                ("units", "peak_channel_id"),
                ("units", "quality"),
                ("units", "firing_rate"),
                ("general", "extracellular_ephys", "electrodes", "id"),
                ("general", "extracellular_ephys", "electrodes", "location"),
            ]

            for path_tuple in required_paths:
                obj = f
                for key in path_tuple:
                    if key not in obj:
                        result["error"] = f"Missing NWB field: {'/'.join(path_tuple)}"
                        return result
                    obj = obj[key]

            unit_ids_all = f["units"]["id"][()]
            peak_channel_ids = f["units"]["peak_channel_id"][()]
            unit_qualities = f["units"]["quality"][()]
            firing_rates = f["units"]["firing_rate"][()]

            electrode_ids = f["general"]["extracellular_ephys"]["electrodes"]["id"][()]
            locations_raw = f["general"]["extracellular_ephys"]["electrodes"]["location"][()]

            locations = [
                loc.decode("utf-8") if isinstance(loc, bytes) else str(loc)
                for loc in locations_raw
            ]

            elec_dict = dict(zip(electrode_ids, locations))

            selected_unit_indices = []
            selected_unit_regions = []

            for row_idx, p_chan in enumerate(peak_channel_ids):
                region = elec_dict.get(p_chan)

                quality = unit_qualities[row_idx]
                if isinstance(quality, bytes):
                    is_good = quality == b"good"
                else:
                    is_good = str(quality) == "good"

                if (
                    region in target_regions
                    and is_good
                    and firing_rates[row_idx] > 0.5
                ):
                    selected_unit_indices.append(row_idx)
                    selected_unit_regions.append(region)

            selected_unit_ids = [
                int(unit_ids_all[idx]) for idx in selected_unit_indices
            ]

            num_presentations = len(valid_ids)
            num_units = len(selected_unit_ids)

            # =========================
            # 4. The full matrix does not need to be materialized
            #    Summarize sizes only to save memory
            # =========================
            region_counts_raw = pd.Series(selected_unit_regions).value_counts().to_dict()

            region_counts = {
                region: int(region_counts_raw.get(region, 0))
                for region in region_order
            }

            result["status"] = "success"
            result["num_presentations"] = int(num_presentations)
            result["num_units"] = int(num_units)
            result["design_matrix_shape"] = [int(num_presentations), int(num_units)]
            result["targets_shape"] = [int(targets.shape[0])]
            result["num_image_classes"] = int(targets.nunique())
            result["region_unit_counts"] = {
                str(k): int(v) for k, v in region_counts.items()
            }

            return result

    except Exception as e:
        result["error"] = repr(e)
        return result


# =========================
# 5. Iterate over all sessions
# =========================
session_dirs = [
    os.path.join(cache_dir, name)
    for name in os.listdir(cache_dir)
    if name.startswith("session_") and os.path.isdir(os.path.join(cache_dir, name))
]

session_dirs = sorted(session_dirs)

print(f"Found {len(session_dirs)} session directories")
print(f"Target regions: {target_regions}")

all_results = []

for session_dir in tqdm(session_dirs, desc="Processing sessions"):
    res = process_one_session(session_dir, target_regions)
    if res is not None:
        all_results.append(res)

# =========================
# 6. Save JSON
# =========================
summary = {
    "cache_dir": cache_dir,
    "target_regions": target_regions,
    "num_sessions_found": len(session_dirs),
    "num_success": sum(1 for x in all_results if x["status"] == "success"),
    "num_failed": sum(1 for x in all_results if x["status"] == "failed"),
    "results": all_results,
}

# with open(output_json, "w", encoding="utf-8") as f:
#     json.dump(summary, f, ensure_ascii=False, indent=4)

print("\nSummary complete.")
print(f"Successfully read: {summary['num_success']}")
print(f"Failed to read: {summary['num_failed']}")
# print(f"Results saved to: {output_json}")

# =========================
# 7. Console summary
# =========================
print("\n===== Session Matrix Size Summary =====")
rows = []

for item in all_results:
    if item["status"] == "success":
        regions = item["region_unit_counts"]

        rows.append({
            "session_id": item["session_id"],
            "session_type": item.get("session_type", "unknown"),
            "X_shape": item["design_matrix_shape"],
            "y_shape": item["targets_shape"],
            "classes": item["num_image_classes"],
            "VISp": regions.get("VISp", 0),
            "VISl": regions.get("VISl", 0),
            "VISrl": regions.get("VISrl", 0),
            "total_units": item["num_units"],
        })
    else:
        rows.append({
            "session_id": item["session_id"],
            "session_type": item.get("session_type", "unknown"),
            "X_shape": "failed",
            "y_shape": "failed",
            "classes": "failed",
            "VISp": 0,
            "VISl": 0,
            "VISrl": 0,
            "total_units": 0,
            "error": item["error"],
        })

df_summary = pd.DataFrame(rows)

print("\n===== Session Matrix Size Summary Table =====")
print(df_summary.to_string(index=False))
