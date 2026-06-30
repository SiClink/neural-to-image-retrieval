import numpy as np
import torch
from torch.utils.data import Dataset


class MultiSessionRetrievalTokenDataset(Dataset):
    def __init__(self, sessions, index_list, raw_label_to_feature_row):
        self.sessions = sessions
        self.index_list = list(index_list)
        self.raw_label_to_feature_row = raw_label_to_feature_row
        print(f"Dataset trials: {len(self.index_list)}")

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        sess_idx, trial_idx = self.index_list[idx]
        sess = self.sessions[sess_idx]

        response = sess["spike_count_selected_norm"][trial_idx]  # [N_sess]
        ccf = sess["ccf_selected_norm"]                          # [N_sess, 3]
        region_ids = sess["region_ids_selected"]                 # [N_sess]

        raw_label = int(sess["labels"][trial_idx])
        feature_row = self.raw_label_to_feature_row[raw_label]

        return {
            "response": torch.tensor(response, dtype=torch.float32),
            "ccf": torch.tensor(ccf, dtype=torch.float32),
            "region_ids": torch.tensor(region_ids, dtype=torch.long),
            "raw_label": torch.tensor(raw_label, dtype=torch.long),
            "feature_row": torch.tensor(feature_row, dtype=torch.long),
            "session_idx": torch.tensor(sess_idx, dtype=torch.long),
            "trial_idx": torch.tensor(trial_idx, dtype=torch.long),
        }


def collate_fn(batch):
    batch_size = len(batch)
    max_units = max(item["response"].shape[0] for item in batch)

    responses = torch.zeros(batch_size, max_units, dtype=torch.float32)
    ccf = torch.zeros(batch_size, max_units, 3, dtype=torch.float32)
    region_ids = torch.zeros(batch_size, max_units, dtype=torch.long)
    mask = torch.zeros(batch_size, max_units, dtype=torch.bool)

    raw_labels = torch.zeros(batch_size, dtype=torch.long)
    feature_rows = torch.zeros(batch_size, dtype=torch.long)
    session_idx = torch.zeros(batch_size, dtype=torch.long)
    trial_idx = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["response"].shape[0]
        responses[i, :n] = item["response"]
        ccf[i, :n] = item["ccf"]
        region_ids[i, :n] = item["region_ids"]
        mask[i, :n] = True
        raw_labels[i] = item["raw_label"]
        feature_rows[i] = item["feature_row"]
        session_idx[i] = item["session_idx"]
        trial_idx[i] = item["trial_idx"]

    return {
        "response": responses,      # [B, max_N]
        "ccf": ccf,                 # [B, max_N, 3]
        "region_ids": region_ids,   # [B, max_N]
        "mask": mask,               # [B, max_N], True=real unit, False=padding
        "raw_label": raw_labels,
        "feature_row": feature_rows,
        "session_idx": session_idx,
        "trial_idx": trial_idx,
    }


class FrameBalancedBatchSampler:
    """
    Training sampler:
        pick frames_per_batch labels, then trials_per_frame trials per label.

    This creates multiple trials per label in a batch for label-centroid loss.
    """

    def __init__(
        self,
        dataset,
        frames_per_batch=72,
        trials_per_frame=1,
        steps_per_epoch=200,
        seed=42,
        drop_last=False,
    ):
        self.dataset = dataset
        self.frames_per_batch = int(frames_per_batch)
        self.trials_per_frame = int(trials_per_frame)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self.label_to_indices = {}
        for ds_idx, (sess_idx, trial_idx) in enumerate(dataset.index_list):
            raw_label = int(dataset.sessions[sess_idx]["labels"][trial_idx])
            self.label_to_indices.setdefault(raw_label, []).append(ds_idx)

        self.labels = np.asarray(sorted(self.label_to_indices.keys()), dtype=np.int64)

        if len(self.labels) == 0:
            raise ValueError("FrameBalancedBatchSampler: no labels are available")

        if self.frames_per_batch > len(self.labels):
            print(
                f"Warning: frames_per_batch={self.frames_per_batch} exceeds the number of available labels={len(self.labels)}; "
                f"using all labels instead."
            )
            self.frames_per_batch = len(self.labels)

        if self.trials_per_frame <= 0:
            raise ValueError("trials_per_frame must be > 0")

        if self.steps_per_epoch <= 0:
            batch_size = self.frames_per_batch * self.trials_per_frame
            self.steps_per_epoch = max(1, len(dataset) // max(batch_size, 1))

        print(
            "FrameBalancedBatchSampler: "
            f"labels={len(self.labels)}, "
            f"frames_per_batch={self.frames_per_batch}, "
            f"trials_per_frame={self.trials_per_frame}, "
            f"batch_size={self.frames_per_batch * self.trials_per_frame}, "
            f"steps_per_epoch={self.steps_per_epoch}"
        )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        for _ in range(self.steps_per_epoch):
            chosen_labels = rng.choice(
                self.labels,
                size=self.frames_per_batch,
                replace=False,
            )

            batch_indices = []
            for lab in chosen_labels:
                candidates = self.label_to_indices[int(lab)]
                replace = len(candidates) < self.trials_per_frame
                chosen = rng.choice(
                    np.asarray(candidates, dtype=np.int64),
                    size=self.trials_per_frame,
                    replace=replace,
                )
                batch_indices.extend(chosen.astype(int).tolist())

            rng.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.steps_per_epoch
