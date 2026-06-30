import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import config_to_dict
from training.centroid import centroid_retrieval_loss, label_centroids_from_batch
from training.checkpointing import (
    BEST_CHECKPOINT_NAME,
    LAST_CHECKPOINT_NAME,
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)
from training.losses import (
    regression_alignment_loss,
    topk_metrics_from_logits,
)
from utils.experiment_writer import (
    BEST_VAL_PREDICTIONS_NAME,
    FINAL_TEST_PREDICTIONS_NAME,
    write_history_json,
    write_prediction_rows_csv,
    write_predictions_npz,
)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    img_features_all,
    text_features_all,
    train_candidate_rows,
    train_raw_to_pos,
    loss_config,
):
    """
    Train with trial embeddings -> batch label centroids -> prototype retrieval.
    """
    model.train()

    img_features_all = F.normalize(img_features_all.to(device).float(), dim=-1)

    if text_features_all is not None and loss_config.alpha < 1.0:
        text_features_all = F.normalize(text_features_all.to(device).float(), dim=-1)
    else:
        text_features_all = None

    train_candidate_rows_t = torch.tensor(train_candidate_rows, dtype=torch.long, device=device)
    img_candidate = img_features_all[train_candidate_rows_t]

    if text_features_all is not None:
        text_candidate = text_features_all[train_candidate_rows_t]
    else:
        text_candidate = None

    total_loss = 0.0
    total_ce_img = 0.0
    total_ce_text = 0.0
    total_cos = 0.0
    total_mse = 0.0
    total_centroids = 0

    top1_sum = 0.0
    top5_sum = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        response = batch["response"].to(device)
        ccf = batch["ccf"].to(device)
        region_ids = batch["region_ids"].to(device)
        mask = batch["mask"].to(device)

        raw_labels = batch["raw_label"].to(device)
        feature_rows = batch["feature_row"].to(device)

        optimizer.zero_grad()

        trial_features = model(response, ccf, region_ids, mask)  # [B, D]
        trial_features = F.normalize(trial_features, dim=-1)

        centroid_features, centroid_labels, centroid_rows = label_centroids_from_batch(
            trial_features,
            raw_labels,
            feature_rows,
        )
        centroid_features = F.normalize(centroid_features, dim=-1)  # [C, D]

        target_img = img_features_all[centroid_rows]
        logit_scale = model.logit_scale.exp().clamp(max=100)

        img_ce, logits_img, target_pos = centroid_retrieval_loss(
            centroid_features=centroid_features,
            centroid_labels=centroid_labels,
            candidate_features=img_candidate,
            raw_to_pos=train_raw_to_pos,
            logit_scale=logit_scale,
        )

        if text_candidate is not None:
            text_ce, _, _ = centroid_retrieval_loss(
                centroid_features=centroid_features,
                centroid_labels=centroid_labels,
                candidate_features=text_candidate,
                raw_to_pos=train_raw_to_pos,
                logit_scale=logit_scale,
            )
        else:
            text_ce = torch.zeros([], device=device)

        cos_loss, mse_loss = regression_alignment_loss(centroid_features, target_img)

        ce_loss = loss_config.alpha * img_ce + (1.0 - loss_config.alpha) * text_ce
        loss = (
            loss_config.ce_weight * ce_loss
            + loss_config.cosine_weight * cos_loss
            + loss_config.mse_weight * mse_loss
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        c = centroid_features.shape[0]
        total_centroids += c

        total_loss += loss.item() * c
        total_ce_img += img_ce.item() * c
        total_ce_text += text_ce.item() * c
        total_cos += cos_loss.item() * c
        total_mse += mse_loss.item() * c

        metrics = topk_metrics_from_logits(logits_img.detach(), target_pos.detach(), ks=(1, 5))
        top1_sum += metrics["top1"] * c
        top5_sum += metrics["top5"] * c

    return {
        "loss": total_loss / total_centroids,
        "img_ce": total_ce_img / total_centroids,
        "text_ce": total_ce_text / total_centroids,
        "cos_loss": total_cos / total_centroids,
        "mse_loss": total_mse / total_centroids,
        "top1": top1_sum / total_centroids,
        "top5": top5_sum / total_centroids,
    }


@torch.no_grad()
def evaluate_retrieval(
    model,
    loader,
    device,
    img_features_all,
    text_features_all,
    candidate_rows,
    candidate_raw_labels,
    candidate_raw_to_pos,
    loss_config,
    name="Eval",
):
    """
    Evaluate by averaging all trial embeddings of the same raw label before retrieval.
    """
    model.eval()

    img_features_all = F.normalize(img_features_all.to(device).float(), dim=-1)

    if text_features_all is not None and loss_config.alpha < 1.0:
        text_features_all = F.normalize(text_features_all.to(device).float(), dim=-1)
    else:
        text_features_all = None

    candidate_rows_t = torch.tensor(candidate_rows, dtype=torch.long, device=device)
    img_candidate = img_features_all[candidate_rows_t]

    if text_features_all is not None:
        text_candidate = text_features_all[candidate_rows_t]
    else:
        text_candidate = None

    all_trial_features = []
    all_raw_labels = []
    all_feature_rows = []
    all_session_idx = []
    all_trial_idx = []

    for batch in tqdm(loader, desc=name, leave=False):
        response = batch["response"].to(device)
        ccf = batch["ccf"].to(device)
        region_ids = batch["region_ids"].to(device)
        mask = batch["mask"].to(device)

        trial_features = model(response, ccf, region_ids, mask)
        trial_features = F.normalize(trial_features, dim=-1)

        all_trial_features.append(trial_features.detach().cpu())
        all_raw_labels.append(batch["raw_label"].detach().cpu())
        all_feature_rows.append(batch["feature_row"].detach().cpu())
        all_session_idx.append(batch["session_idx"].detach().cpu())
        all_trial_idx.append(batch["trial_idx"].detach().cpu())

    all_trial_features = torch.cat(all_trial_features, dim=0).to(device)
    all_raw_labels = torch.cat(all_raw_labels, dim=0).to(device)
    all_feature_rows = torch.cat(all_feature_rows, dim=0).to(device)

    all_session_idx_np = torch.cat(all_session_idx, dim=0).numpy().astype(np.int64)
    all_trial_idx_np = torch.cat(all_trial_idx, dim=0).numpy().astype(np.int64)

    centroid_features, centroid_labels, centroid_rows = label_centroids_from_batch(
        all_trial_features,
        all_raw_labels,
        all_feature_rows,
    )
    centroid_features = F.normalize(centroid_features, dim=-1)

    target_img = img_features_all[centroid_rows]
    logit_scale = model.logit_scale.exp().clamp(max=100)

    img_ce, logits_img, target_pos = centroid_retrieval_loss(
        centroid_features=centroid_features,
        centroid_labels=centroid_labels,
        candidate_features=img_candidate,
        raw_to_pos=candidate_raw_to_pos,
        logit_scale=logit_scale,
    )

    if text_candidate is not None:
        text_ce, _, _ = centroid_retrieval_loss(
            centroid_features=centroid_features,
            centroid_labels=centroid_labels,
            candidate_features=text_candidate,
            raw_to_pos=candidate_raw_to_pos,
            logit_scale=logit_scale,
        )
    else:
        text_ce = torch.zeros([], device=device)

    cos_loss, mse_loss = regression_alignment_loss(centroid_features, target_img)

    ce_loss = loss_config.alpha * img_ce + (1.0 - loss_config.alpha) * text_ce
    loss = (
        loss_config.ce_weight * ce_loss
        + loss_config.cosine_weight * cos_loss
        + loss_config.mse_weight * mse_loss
    )

    metrics = topk_metrics_from_logits(logits_img.detach(), target_pos.detach(), ks=(1, 5))

    pred_pos = torch.argmax(logits_img, dim=1).detach().cpu().numpy()
    pred_raw = candidate_raw_labels[pred_pos]

    return {
        "loss": float(loss.item()),
        "img_ce": float(img_ce.item()),
        "text_ce": float(text_ce.item()),
        "cos_loss": float(cos_loss.item()),
        "mse_loss": float(mse_loss.item()),
        "top1": float(metrics["top1"]),
        "top5": float(metrics["top5"]),
        "mean_rank": float(metrics["mean_rank"]),
        "median_rank": float(metrics["median_rank"]),
        "pred_raw_labels": pred_raw.astype(np.int64),
        "true_raw_labels": centroid_labels.detach().cpu().numpy().astype(np.int64),
        "feature_rows": centroid_rows.detach().cpu().numpy().astype(np.int64),
        "centroid_features": centroid_features.detach().cpu().numpy().astype(np.float32),
        "trial_raw_labels": all_raw_labels.detach().cpu().numpy().astype(np.int64),
        "session_idx": all_session_idx_np,
        "trial_idx": all_trial_idx_np,
        "spike_features": all_trial_features.detach().cpu().numpy().astype(np.float32),
    }


def run_training(
    model,
    train_loader,
    train_batch_sampler,
    val_loader,
    test_loader,
    optimizer,
    scheduler,
    device,
    img_features_all,
    text_features_all,
    candidates,
    cfg,
    save_dir,
    input_unit_stats,
    proj_dim,
    session_ids,
    target_regions,
    raw_label_to_feature_row,
    args_dict,
):
    best_val_top1 = 0.0
    best_epoch = 0
    bad_epochs = 0
    history = []

    best_path = os.path.join(save_dir, BEST_CHECKPOINT_NAME)
    last_path = os.path.join(save_dir, LAST_CHECKPOINT_NAME)
    config_dict = config_to_dict(cfg)

    print("\n================ Starting centroid retrieval training ================")

    for epoch in range(1, cfg.optim.epochs + 1):
        train_batch_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            img_features_all=img_features_all,
            text_features_all=text_features_all,
            train_candidate_rows=candidates["train"]["candidate_rows"],
            train_raw_to_pos=candidates["train"]["raw_to_pos"],
            loss_config=cfg.loss,
        )

        val_metrics = evaluate_retrieval(
            model=model,
            loader=val_loader,
            device=device,
            img_features_all=img_features_all,
            text_features_all=text_features_all,
            candidate_rows=candidates["val"]["candidate_rows"],
            candidate_raw_labels=candidates["val"]["raw_labels"],
            candidate_raw_to_pos=candidates["val"]["raw_to_pos"],
            loss_config=cfg.loss,
            name="Val",
        )

        scheduler.step()

        print(
            f"Epoch [{epoch:03d}/{cfg.optim.epochs}] "
            f"| Train Loss: {train_metrics['loss']:.4f} "
            f"| Train Top1: {train_metrics['top1']:.4f} "
            f"| Train Top5: {train_metrics['top5']:.4f} "
            f"| Val Loss: {val_metrics['loss']:.4f} "
            f"| Val Top1: {val_metrics['top1']:.4f} "
            f"| Val Top5: {val_metrics['top5']:.4f} "
            f"| Val MeanRank: {val_metrics['mean_rank']:.2f}"
        )

        history_row = {
            "epoch": int(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": {k: float(v) for k, v in train_metrics.items()},
            "val": {
                k: float(v)
                for k, v in val_metrics.items()
                if isinstance(v, (int, float, np.integer, np.floating))
            },
        }
        history.append(history_row)
        write_history_json(save_dir, history)

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = float(val_metrics["top1"])
            best_epoch = epoch
            bad_epochs = 0

            payload = build_checkpoint_payload(
                model=model,
                model_config=cfg.model,
                session_ids=session_ids,
                target_regions=target_regions,
                train_labels=candidates["train"]["raw_labels"],
                val_labels=candidates["val"]["raw_labels"],
                test_labels=candidates["test"]["raw_labels"],
                train_candidate_rows=candidates["train"]["candidate_rows"],
                val_candidate_rows=candidates["val"]["candidate_rows"],
                test_candidate_rows=candidates["test"]["candidate_rows"],
                raw_label_to_feature_row=raw_label_to_feature_row,
                input_unit_stats=input_unit_stats,
                proj_dim=proj_dim,
                args_dict=args_dict,
                config_dict=config_dict,
                best_val_top1=best_val_top1,
                best_epoch=best_epoch,
            )
            save_checkpoint(payload, best_path)
            write_predictions_npz(save_dir, BEST_VAL_PREDICTIONS_NAME, val_metrics)
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.optim.patience:
            print(f"Early stopping: validation top1 did not improve for {cfg.optim.patience} consecutive epochs")
            break

    last_payload = build_checkpoint_payload(
        model=model,
        model_config=cfg.model,
        session_ids=session_ids,
        target_regions=target_regions,
        train_labels=candidates["train"]["raw_labels"],
        val_labels=candidates["val"]["raw_labels"],
        test_labels=candidates["test"]["raw_labels"],
        train_candidate_rows=candidates["train"]["candidate_rows"],
        val_candidate_rows=candidates["val"]["candidate_rows"],
        test_candidate_rows=candidates["test"]["candidate_rows"],
        raw_label_to_feature_row=raw_label_to_feature_row,
        input_unit_stats=input_unit_stats,
        proj_dim=proj_dim,
        args_dict=args_dict,
        config_dict=config_dict,
        best_val_top1=best_val_top1,
        best_epoch=best_epoch,
    )
    save_checkpoint(last_payload, last_path)

    print("================ Training finished ================")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Top1: {best_val_top1:.4f}")

    if os.path.exists(best_path):
        checkpoint = load_checkpoint(best_path, map_location=device)
    else:
        checkpoint = load_checkpoint(last_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_test_metrics = evaluate_retrieval(
        model=model,
        loader=test_loader,
        device=device,
        img_features_all=img_features_all,
        text_features_all=text_features_all,
        candidate_rows=candidates["test"]["candidate_rows"],
        candidate_raw_labels=candidates["test"]["raw_labels"],
        candidate_raw_to_pos=candidates["test"]["raw_to_pos"],
        loss_config=cfg.loss,
        name="FinalTest",
    )

    print(
        f"\nFinal Test Loss: {final_test_metrics['loss']:.4f} "
        f"| Final Test Top1: {final_test_metrics['top1']:.4f} "
        f"| Final Test Top5: {final_test_metrics['top5']:.4f} "
        f"| Final Test MeanRank: {final_test_metrics['mean_rank']:.2f}"
    )

    final_predictions_path = write_predictions_npz(
        save_dir,
        FINAL_TEST_PREDICTIONS_NAME,
        final_test_metrics,
    )
    csv_path = write_prediction_rows_csv(save_dir, final_test_metrics)

    return {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_top1": float(best_val_top1),
        "best_path": best_path,
        "last_path": last_path,
        "final_test_metrics": final_test_metrics,
        "final_test_predictions_path": final_predictions_path,
        "final_test_prediction_rows_csv": csv_path,
    }
