import json
import os
import zipfile

import numpy as np
from numpy.lib import format as np_format


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path, obj, indent=4):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=json_default)
    return path


def save_npz(path, **arrays):
    """
    Save a standard .npz archive without calling np.savez directly.

    Some NumPy builds can fail inside np.savez dispatch even for plain ndarray
    inputs. Writing each .npy member into the zip keeps np.load-compatible output
    while avoiding that dispatch path.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as zip_file:
        for key, value in arrays.items():
            if not isinstance(key, str) or key == "":
                raise ValueError(f"npz key must be a non-empty string, got: {key!r}")
            if key.endswith(".npy") or "/" in key or "\\" in key:
                raise ValueError(f"npz key cannot contain path separators or a .npy suffix, got: {key!r}")

            array = np.asanyarray(value)
            with zip_file.open(f"{key}.npy", mode="w", force_zip64=True) as file_obj:
                np_format.write_array(file_obj, array, allow_pickle=True)
    return path
