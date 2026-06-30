import os
import h5py
import numpy as np
import pandas as pd


# ============================================================
# Centralized path settings
# ============================================================

# selected_sessions directory
selected_dir = os.environ.get("SELECTED_SESSIONS_DIR", "data/selected_sessions")

# Allen ecephys cache root directory
cache_dir = os.environ.get("ALLEN_ECEPHYS_CACHE_DIR", "data/ecephys_cache_dir")

# CSV file paths
units_csv_path = os.path.join(cache_dir, "units.csv")
channels_csv_path = os.path.join(cache_dir, "channels.csv")

# Target brain regions
target_regions = ["VISp", "VISl", "VISrl"]

# Unit selection criteria
min_firing_rate = 0.5


# ============================================================
# Helper functions
# ============================================================

def decode_region_array(region_arr):
    regions = []
    for r in region_arr:
        if isinstance(r, bytes):
            regions.append(r.decode("utf-8"))
        else:
            regions.append(str(r))
    return np.array(regions)


def get_session_id_from_filename(filename):
    """
    Example:
    732592105_selected.npz -> 732592105
    """
    return int(filename.replace("_selected.npz", ""))


# ============================================================
# Read CSV files and build the unit_id -> CCF + region mapping
# ============================================================

print("=" * 80)
print("Reading units.csv and channels.csv")
print("=" * 80)

units_table = pd.read_csv(units_csv_path)
channels_table = pd.read_csv(channels_csv_path)

channel_to_info = {}

for _, row in channels_table.iterrows():
    channel_id = int(row["id"])

    ccf = [
        row["anterior_posterior_ccf_coordinate"],
        row["dorsal_ventral_ccf_coordinate"],
        row["left_right_ccf_coordinate"]
    ]

    ccf = [float(x) if pd.notna(x) else np.nan for x in ccf]

    region = row["ecephys_structure_acronym"]
    region = "" if pd.isna(region) else str(region)

    channel_to_info[channel_id] = {
        "ccf": ccf,
        "region": region
    }


unit_to_info = {}

for _, row in units_table.iterrows():
    unit_id = int(row["id"])
    channel_id = int(row["ecephys_channel_id"])

    if channel_id in channel_to_info:
        unit_to_info[unit_id] = channel_to_info[channel_id]


# ============================================================
# Part 1: scan selected_sessions and build the summary table
# ============================================================

npz_files = sorted([
    f for f in os.listdir(selected_dir)
    if f.endswith("_selected.npz")
])

summary_rows = []
missing_session_ids = []

for filename in npz_files:
    session_id = get_session_id_from_filename(filename)
    npz_path = os.path.join(selected_dir, filename)

    data = np.load(npz_path, allow_pickle=True)

    if "ccf" not in data:
        summary_rows.append({
            "session_id": session_id,
            "file": filename,
            "total_units": None,
            "valid_ccf_units": None,
            "missing_ccf_units": None,
            "missing_ratio": None,
            "VISp": None,
            "VISl": None,
            "VISrl": None,
            "has_ccf_field": False
        })
        missing_session_ids.append(session_id)
        continue

    ccf = data["ccf"]
    total_units = ccf.shape[0]

    invalid_mask = ~np.isfinite(ccf)
    unit_missing_mask = invalid_mask.any(axis=1)

    missing_ccf_units = int(unit_missing_mask.sum())
    valid_ccf_units = int(total_units - missing_ccf_units)

    if missing_ccf_units > 0:
        missing_session_ids.append(session_id)

    region_count = {r: 0 for r in target_regions}

    if "region" in data:
        region = decode_region_array(data["region"])

        if len(region) == total_units:
            for r in target_regions:
                region_count[r] = int((region == r).sum())

    missing_ratio = missing_ccf_units / total_units if total_units > 0 else 0

    summary_rows.append({
        "session_id": session_id,
        "file": filename,
        "total_units": total_units,
        "valid_ccf_units": valid_ccf_units,
        "missing_ccf_units": missing_ccf_units,
        "missing_ratio": f"{missing_ratio * 100:.2f}%",
        "VISp": region_count["VISp"],
        "VISl": region_count["VISl"],
        "VISrl": region_count["VISrl"],
        "has_ccf_field": True
    })


summary_df = pd.DataFrame(summary_rows)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


print("\n" + "=" * 80)
print("selected_sessions summary table")
print("=" * 80)

print(summary_df.to_string(index=False))


# ============================================================
# Overall summary
# ============================================================

total_files = len(summary_df)
total_units_all = summary_df["total_units"].dropna().astype(int).sum()
total_valid_all = summary_df["valid_ccf_units"].dropna().astype(int).sum()
total_missing_all = summary_df["missing_ccf_units"].dropna().astype(int).sum()
files_with_missing = (summary_df["missing_ccf_units"].fillna(1).astype(int) > 0).sum()

print("\n" + "=" * 80)
print("Overall statistics")
print("=" * 80)
print(f"Total selected npz files    : {total_files}")
print(f"Sessions with missing CCF   : {files_with_missing}")
print(f"Total selected units        : {total_units_all}")
print(f"Units with valid CCF        : {total_valid_all}")
print(f"Units with missing CCF      : {total_missing_all}")

if total_units_all > 0:
    print(f"Missing CCF ratio           : {total_missing_all / total_units_all * 100:.4f}%")


# ============================================================
# Part 2: inspect original NWB and CSV data for sessions with missing CCF
# ============================================================

missing_session_ids = sorted(list(set(missing_session_ids)))

print("\n" + "=" * 80)
print("Sessions that require original NWB inspection")
print("=" * 80)
print(missing_session_ids)


def inspect_original_nwb_session(session_id):
    nwb_path = os.path.join(
        cache_dir,
        f"session_{session_id}",
        f"session_{session_id}.nwb"
    )

    if not os.path.exists(nwb_path):
        print("\n" + "=" * 60)
        print(f"Session ID: {session_id}")
        print(f"NWB file does not exist: {nwb_path}")
        return

    with h5py.File(nwb_path, "r") as f:
        unit_ids_all = f["units"]["id"][()]
        unit_qualities = f["units"]["quality"][()]
        firing_rates = f["units"]["firing_rate"][()]

        total_target_units = 0
        missing_ccf_units = []
        valid_ccf_units = []

        region_count = {r: 0 for r in target_regions}
        missing_region_count = {r: 0 for r in target_regions}

        for idx, uid in enumerate(unit_ids_all):
            unit_id = int(uid)

            quality = unit_qualities[idx]
            if isinstance(quality, (bytes, np.bytes_)):
                quality = quality.decode("utf-8")
            else:
                quality = str(quality)

            firing_rate = float(firing_rates[idx])

            if quality != "good":
                continue

            if firing_rate <= min_firing_rate:
                continue

            if unit_id not in unit_to_info:
                continue

            region = unit_to_info[unit_id]["region"]
            ccf = np.asarray(unit_to_info[unit_id]["ccf"], dtype=np.float32)

            if region not in target_regions:
                continue

            total_target_units += 1
            region_count[region] += 1

            if np.isfinite(ccf).all():
                valid_ccf_units.append(unit_id)
            else:
                missing_ccf_units.append(unit_id)
                missing_region_count[region] += 1

    print("\n" + "=" * 60)
    print(f"Session ID: {session_id}")
    print(f"Target regions: {target_regions}")
    print(f"Selection criteria: quality == good, firing_rate > {min_firing_rate}")
    print("=" * 60)

    print("Total target-region units:", total_target_units)
    print("Units with valid CCF:", len(valid_ccf_units))
    print("Units with missing CCF:", len(missing_ccf_units))

    print("\nUnit counts by region:")
    for r in target_regions:
        print(f"{r}: {region_count[r]}")

    print("\nMissing CCF counts by region:")
    for r in target_regions:
        print(f"{r}: {missing_region_count[r]}")

    print("\nFirst 20 unit_ids with missing CCF:")
    print(missing_ccf_units[:20])

    print("\nFirst 20 unit_ids with valid CCF:")
    print(valid_ccf_units[:20])


for session_id in missing_session_ids:
    inspect_original_nwb_session(session_id)