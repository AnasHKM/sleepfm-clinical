# trainer_epoch_bars.py
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score
from tqdm.auto import tqdm
import wandb
from models.models import SetTransformer
from utils import load_data


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unwrap(m):
    return m.module if hasattr(m, "module") else m

def freeze_encoder(model):
    enc = unwrap(model.encoder)
    for p in enc.parameters():
        p.requires_grad = False
    enc.eval()

def unfreeze_encoder(model):
    enc = unwrap(model.encoder)
    for p in enc.parameters():
        p.requires_grad = True
    enc.train()

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        num_classes: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        save_path: str = "best_finetune.pt",
        project: str = "sleep-finetune",
        run_name: Optional[str] = None,
        extra_wandb_config: Optional[Dict[str, Any]] = None,
        wandb_mode: str = "offline",   # "offline" if you prefer
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        self.best_val_auc = -np.inf
        self.save_path = save_path

        # ----- Weights & Biases -----
        cfg = dict(
            lr=lr,
            weight_decay=weight_decay,
            num_classes=num_classes,
            batch_size=getattr(train_loader, "batch_size", None),
            model=str(self.model.__class__.__name__),
        )
        if extra_wandb_config:
            cfg.update(extra_wandb_config)
        wandb.init(project=project, name=run_name, config=cfg, mode=wandb_mode)

    @staticmethod
    def _macro_auc_ovr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Safe macro AUC (OVR). Returns NaN if not computable."""
        try:
            # need at least 2 classes present in y_true
            if len(np.unique(y_true)) < 2:
                return float("nan")
            return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            return float("nan")

    def _epoch_forward(self, loader: DataLoader, train: bool):
        """
        Runs one full pass over loader.
        Returns aggregated metrics dict.
        """
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss, total_n = 0.0, 0
        y_true_list, y_prob_list, y_pred_list = [], [], []

        for file_paths, labels in loader:
            labels = labels.to(self.device)

            # forward
            logits = self.model(file_paths, demo_features=None)  # (B, num_classes)
            loss = self.criterion(logits, labels)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                y_true_list.append(labels.detach().cpu().numpy())
                y_prob_list.append(probs.detach().cpu().numpy())
                y_pred_list.append(preds.detach().cpu().numpy())

                total_loss += loss.item() * labels.size(0)
                total_n += labels.size(0)

        avg_loss = total_loss / max(total_n, 1)

        if total_n > 0:
            y_true = np.concatenate(y_true_list, axis=0)
            y_prob = np.concatenate(y_prob_list, axis=0)
            y_pred = np.concatenate(y_pred_list, axis=0)

            auc = self._macro_auc_ovr(y_true, y_prob)
            f1  = f1_score(y_true, y_pred, average="macro")
            rec = recall_score(y_true, y_pred, average="macro")
            acc = accuracy_score(y_true, y_pred)
        else:
            auc = f1 = rec = acc = float("nan")

        return {
            "loss": avg_loss,
            "auc_macro": auc,
            "f1_macro": f1,
            "recall_macro": rec,
            "accuracy": acc,
        }

    def fit(self, epochs: int = 10, validate_every: int = 1):
        # Two persistent bars that advance once per epoch
        train_bar = tqdm(total=epochs, desc="Train (per-epoch)", position=0, leave=True, dynamic_ncols=True)
        val_bar   = tqdm(total=epochs, desc="Val   (per-epoch)", position=1, leave=True, dynamic_ncols=True)

        for ep in range(1, epochs + 1):
            t0 = time.time()

            # ---- Train epoch ----
            tr = self._epoch_forward(self.train_loader, train=True)
            wandb.log({
                "train/loss": tr["loss"],
                "train/auc_macro": tr["auc_macro"],
                "train/f1_macro": tr["f1_macro"],
                "train/recall_macro": tr["recall_macro"],
                "train/accuracy": tr["accuracy"],
                "epoch": ep,
            })

            # Update the train bar exactly once for this epoch
            train_bar.update(1)
            train_bar.set_postfix({
                "loss": f"{tr['loss']:.4f}",
                "auc": f"{tr['auc_macro']:.3f}",
                "f1": f"{tr['f1_macro']:.3f}",
                "rec": f"{tr['recall_macro']:.3f}",
                "acc": f"{tr['accuracy']:.3f}",
            })

            # ---- Validation (optional each epoch) ----
            if ep % validate_every == 0:
                va = self._epoch_forward(self.val_loader, train=False)
                wandb.log({
                    "val/loss": va["loss"],
                    "val/auc_macro": va["auc_macro"],
                    "val/f1_macro": va["f1_macro"],
                    "val/recall_macro": va["recall_macro"],
                    "val/accuracy": va["accuracy"],
                    "epoch": ep,
                })

                # Update the val bar once for this epoch
                val_bar.update(1)
                val_bar.set_postfix({
                    "loss": f"{va['loss']:.4f}",
                    "auc": f"{va['auc_macro']:.3f}",
                    "f1": f"{va['f1_macro']:.3f}",
                    "rec": f"{va['recall_macro']:.3f}",
                    "acc": f"{va['accuracy']:.3f}",
                })

                # Save best by val AUC
                val_auc = va["auc_macro"]
                if not np.isnan(val_auc) and val_auc > self.best_val_auc:
                    self.best_val_auc = val_auc
                    torch.save({"model": self.model.state_dict()}, self.save_path)
                    wandb.log({"model/best_val_auc": self.best_val_auc, "epoch": ep})
            else:
                # still increment the val bar so both bars line up in total steps
                val_bar.update(1)
                val_bar.set_postfix({"loss": "NA", "auc": "NA", "f1": "NA", "rec": "NA", "acc": "NA"})

            dt = time.time() - t0
            wandb.log({"epoch/seconds": dt, "epoch": ep})

        train_bar.close()
        val_bar.close()



if __name__=="__main__":
    from torch.utils.data import DataLoader, Subset
    from sklearn.model_selection import train_test_split
    import numpy as np
    from pipeline.patient_embeddings import CVDDataset, PatientLevelModelLSTMWithDemo, collate_cvddataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_data('./checkpoints/SetTransformer/leave_one_out_128_patch_size_640/config.json')

    encoder = SetTransformer(
        in_channels=config["in_channels"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        pooling_head=config["pooling_head"],
        dropout=0.0,
    )

    encoder = nn.DataParallel(encoder)  # if you used DP for pretrain
    encoder.to(device)

    ckpt = torch.load("./checkpoints/SetTransformer/leave_one_out_128_patch_size_640/best.pt", map_location=device)
    encoder.load_state_dict(ckpt["state_dict"])

    channel_groups = load_data('./configs/channel_groups.json')
    config_data = load_data('./configs/config_finetune_diagnosis_coxph.yaml')

    model = PatientLevelModelLSTMWithDemo(
        encoder=encoder,
        config=config,
        channel_groups=channel_groups,
        num_classes=4,
        device=device,
        chunk_bs=64,
        num_workers=8,
        pooling_head=4,
        num_layers=2,
        dropout=0.3,
    ).to(device)


    patient_dataset = CVDDataset(config_data, channel_groups, Path('/temp_work/ch266186/shhs_hdf5'))

    # Stratified split using the labels you built in index_map
    all_labels = np.array([lbl for _, lbl in patient_dataset.index_map])
    idx = np.arange(len(all_labels))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=all_labels
    )

    train_loader = DataLoader(
        Subset(patient_dataset, train_idx),
        batch_size=4, shuffle=True, num_workers=10, pin_memory=True,
        collate_fn=collate_cvddataset
    )
    val_loader = DataLoader(
        Subset(patient_dataset, val_idx),
        batch_size=2, shuffle=False, num_workers=10, pin_memory=True,
        collate_fn=collate_cvddataset
    )

    freeze_encoder(model)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=4,
        lr=1e-3,
        weight_decay=1e-2,
        save_path="best_finetune.pt",
        project="sleep-finetune",
        run_name="st_lstm_with_demo",
        wandb_mode="online"
    )

    n = count_trainable_params(model)
    print(f"Trainable parameters: {n:,} ({n / 1e6:.2f}M)")

    trainer.fit(epochs=30)
