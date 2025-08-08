import os
from pathlib import Path
import random
from typing import List, Tuple
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# your code
from models.dataset_aws import SetTransformerDataset, collate_fn
from models.models import SetTransformer, DiagnosisFinetuneFullLSTMCOXPHWithDemo
from utils import load_data


class CVDDataset(Dataset):
    def __init__(self,
                 config,
                 channel_groups,
                 hdf5_path,
                 ):

        super().__init__()

        self.config = config
        self.channel_groups = channel_groups
        self.max_channels = self.config["max_channels"]

        hdf5_paths = list(hdf5_path.glob("*.hdf5"))

        self.demo_data = pd.read_csv('./dataset/clinical_features.csv')

        pid_to_label = dict(zip(self.demo_data.nsrrid, self.demo_data.LABEL))


        self.index_map = [(path, pid_to_label[int(path.stem.split('-')[-1])]) for path in hdf5_paths]
        self.total_len = len(self.index_map)
        self.max_seq_len = config["model_params"]["max_seq_length"]

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        hdf5_path, label = self.index_map[idx]

        return hdf5_path, torch.tensor(label).long()

def collate_cvddataset(batch):
    paths = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch])
    return paths, labels


class PatientLevelModelLSTMWithDemo(nn.Module):
    """
    - Uses your SetTransformer encoder to embed 5s chunks (keeps grad — can fine-tune)
    - Collates a full night into (1, C, S, E) + (1, C, S) mask
    - Runs DiagnosisFinetuneFullLSTMCOXPHWithDemo head
    - forward() accepts List[Path] (one or many patients), and demo features
    """
    def __init__(
        self,
        encoder: nn.Module,
        config: dict,
        channel_groups: dict,
        num_classes: int,
        device: torch.device,
        chunk_bs: int = 64,
        num_workers: int = 4,
        pooling_head: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        id_from_path=lambda p: Path(p).stem.split('-')[-1],
    ):
        super().__init__()
        self.encoder = encoder
        self.cfg = config
        self.groups = channel_groups
        self.device = device
        self.chunk_bs = chunk_bs
        self.num_workers = num_workers
        self.id_from_path = id_from_path

        self.modality_types = self.cfg["modality_types"]
        self.M = len(self.modality_types)
        self.embed_dim = 128
        self.num_classes = num_classes

        self.classifier = DiagnosisFinetuneFullLSTMCOXPHWithDemo(
            embed_dim=self.embed_dim,
            num_heads=4,     # not used directly, kept for parity
            num_layers=2,
            num_classes=num_classes,
            pooling_head=4,
            dropout=0.3,
            max_seq_length=6480,                        # let it be large
        )

    @torch.no_grad()
    def _count_chunks_for_patient(self, file_path: Path) -> int:
        ds = SetTransformerDataset(self.cfg, self.groups, hdf5_paths=file_path)
        return len(ds)

    @torch.no_grad()
    def _encode_full_patient(self, file_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          x:    (1, M, T, E)  embeddings for a single patient
          mask: (1, M, T)     0 = real, 1 = pad
        """
        ds = SetTransformerDataset(self.cfg, self.groups, hdf5_paths=file_path)
        loader = DataLoader(
            ds, batch_size=self.chunk_bs, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=False
        )

        # per-modality buffer of chunk embeddings (keep graph / no detach!)
        mod_bufs = [[] for _ in range(self.M)]

        for batch_data, mask_list, file_paths, _, chunk_starts in loader:
            for m_idx in range(self.M):
                x = batch_data[m_idx].to(self.device)            # (B,C,L)
                m = mask_list[m_idx].to(self.device).bool()       # (B,C) 0=real,1=pad
                out = self.encoder(x, m)                          # tuple or tensor
                emb = out[1]  # (B,E)
                mod_bufs[m_idx].append(emb)                       # keep on device

        # concat over time for each modality -> (T,E)
        seqs = [torch.cat(buf, dim=0) if len(buf) > 0 else
                torch.zeros(0, self.embed_dim, device=self.device) for buf in mod_bufs]

        patient_data = torch.stack([t.view(-1, 128) for t in seqs], dim=0)

        max_seq_len = 6480  # or get it from config
        C, T, E = patient_data.shape  # (C, T, 128)

        if T < max_seq_len:
            # Pad time dimension
            pad = torch.zeros((C, max_seq_len - T, E), dtype=patient_data.dtype, device=patient_data.device)
            x = torch.cat([patient_data, pad], dim=1)  # (C, max_seq_len, E)

            mk = torch.cat([
                torch.zeros((C, T), dtype=torch.bool, device=patient_data.device),
                torch.ones((C, max_seq_len - T), dtype=torch.bool, device=patient_data.device)
            ], dim=1)  # (C, max_seq_len)

        else:
            # Truncate time dimension
            x = patient_data[:, :max_seq_len, :]  # (C, max_seq_len, E)
            mk = torch.zeros((C, max_seq_len), dtype=torch.bool, device=patient_data.device)

        # Add batch dimension
        x = x.unsqueeze(0)  # (1, C, max_seq_len, E)
        mk = mk.unsqueeze(0)  # (1, C, max_seq_len)

        return x, mk

    def forward(
        self,
        file_paths,
        demo_features=None,
    ) -> torch.Tensor:
        """
        file_paths: list of patient HDF5 paths
        demo_features:
          - Tensor of shape (B,2), aligned with file_paths, or
          - dict {sid -> tensor(2)}, where sid is parsed by id_from_path
          - None -> zeros
        returns logits: (B, num_classes)
        """
        B = len(file_paths)
        logits = []
        # prepare demo
        if isinstance(demo_features, torch.Tensor):
            assert demo_features.shape == (B, 2), "demo_features must be (B,2)"
            demo_batch = demo_features.to(self.device, dtype=torch.float)
        else:
            demo_batch = None  # we’ll build per-patient if dict/None

        for i, fp in enumerate(file_paths):
            x, mk = self._encode_full_patient(Path(fp))   # (1,M,T,E), (1,M,T)
            if demo_batch is None:
                if isinstance(demo_features, dict):
                    sid = self.id_from_path(fp)
                    d = torch.as_tensor(demo_features.get(sid, [0.0, 0.0]), dtype=torch.float, device=self.device).view(1, 2)
                else:
                    d = torch.zeros(1, 2, device=self.device)
            else:
                d = demo_batch[i].unsqueeze(0)           # (1,2)


            out = self.classifier(x, mk, d)
            logits.append(out)

        return torch.cat(logits, dim=0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_data('../checkpoints/SetTransformer/leave_one_out_128_patch_size_640/config.json')



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

    ckpt = torch.load("../checkpoints/SetTransformer/leave_one_out_128_patch_size_640/best.pt", map_location=device)
    encoder.load_state_dict(ckpt["state_dict"])

    channel_groups = load_data('../configs/channel_groups.json')
    config_data = load_data('../configs/config_finetune_diagnosis_coxph.yaml')

    patient_dataset = CVDDataset(config_data, channel_groups, Path(r'/temp_work/ch266186/shhs_hdf5'))



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

    data_loader = DataLoader(patient_dataset,batch_size=3, shuffle=True, collate_fn=collate_cvddataset)







