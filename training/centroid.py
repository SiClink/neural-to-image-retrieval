import numpy as np
import torch
import torch.nn.functional as F


def build_candidate_tensors(raw_labels, raw_label_to_feature_row):
    raw_labels = np.asarray(raw_labels, dtype=np.int64)
    feature_rows = np.asarray(
        [raw_label_to_feature_row[int(x)] for x in raw_labels],
        dtype=np.int64,
    )
    raw_to_pos = {
        int(raw): int(i)
        for i, raw in enumerate(raw_labels.tolist())
    }
    return raw_labels, feature_rows, raw_to_pos


def label_centroids_from_batch(features, raw_labels, feature_rows=None):
    """
    features:     [B, D]
    raw_labels:   [B]
    feature_rows: [B] or None

    Returns centroid_features [C, D], centroid_labels [C], centroid_rows [C] or None.
    """
    unique_labels = torch.unique(raw_labels, sorted=True)

    centroids = []
    rows = []
    for lab in unique_labels:
        mask = raw_labels == lab
        centroids.append(features[mask].mean(dim=0))
        if feature_rows is not None:
            rows.append(feature_rows[mask][0])

    centroid_features = torch.stack(centroids, dim=0)

    if feature_rows is not None:
        centroid_rows = torch.stack(rows, dim=0)
    else:
        centroid_rows = None

    return centroid_features, unique_labels, centroid_rows


def centroid_retrieval_loss(
    centroid_features,
    centroid_labels,
    candidate_features,
    raw_to_pos,
    logit_scale,
):
    device = centroid_features.device
    target_pos = torch.tensor(
        [raw_to_pos[int(x)] for x in centroid_labels.detach().cpu().numpy().tolist()],
        dtype=torch.long,
        device=device,
    )

    logits = logit_scale * centroid_features @ candidate_features.T
    loss = F.cross_entropy(logits, target_pos)
    return loss, logits, target_pos
