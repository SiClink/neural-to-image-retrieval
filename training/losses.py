import torch
import torch.nn.functional as F


def prototype_retrieval_loss(
    spike_features,
    raw_labels,
    candidate_features,
    raw_to_pos,
    logit_scale,
):
    device = spike_features.device
    target_pos = torch.tensor(
        [raw_to_pos[int(x)] for x in raw_labels.detach().cpu().numpy().tolist()],
        dtype=torch.long,
        device=device,
    )

    logits = logit_scale * spike_features @ candidate_features.T
    loss = F.cross_entropy(logits, target_pos)
    return loss, logits, target_pos


def regression_alignment_loss(spike_features, target_features):
    cos_loss = 1.0 - F.cosine_similarity(spike_features, target_features, dim=-1).mean()
    mse_loss = F.mse_loss(spike_features, target_features)
    return cos_loss, mse_loss


def topk_metrics_from_logits(logits, target_pos, ks=(1, 5)):
    metrics = {}
    ranks = []
    order = torch.argsort(logits, dim=1, descending=True)

    for i in range(logits.shape[0]):
        pos = int((order[i] == target_pos[i]).nonzero(as_tuple=False)[0].item())
        ranks.append(pos + 1)

    ranks = torch.tensor(ranks, device=logits.device)

    for k in ks:
        k = min(int(k), logits.shape[1])
        topk = order[:, :k]
        correct = (topk == target_pos.unsqueeze(1)).any(dim=1).float().mean()
        metrics[f"top{k}"] = float(correct.item())

    metrics["mean_rank"] = float(ranks.float().mean().item())
    metrics["median_rank"] = float(ranks.float().median().item())
    return metrics
