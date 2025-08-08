import random
import multiprocessing
from tqdm import tqdm
import h5py
import numpy as np
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from utils import load_data
from pathlib import Path

def index_file_helper(args):
    file_path, channel_like, chunk_size, channel_groups, modality_types = args
    file_index_map = []
    modality_to_channels = {modality_type: [] for modality_type in modality_types}
    try:
        with h5py.File(file_path, 'r', rdcc_nbytes = 300 * 512 * 8 * 2) as hf:
            dset_names = []
            for dset_name in hf.keys():
                if not channel_like or dset_name in channel_like:
                    if isinstance(hf[dset_name], h5py.Dataset):
                        dset_names.append(dset_name)
                        if dset_name in channel_groups["BAS"]:
                            modality_to_channels["BAS"].append(dset_name)
                        if dset_name in channel_groups["RESP"]:
                            modality_to_channels["RESP"].append(dset_name)
                        if dset_name in channel_groups["EKG"]:
                            modality_to_channels["EKG"].append(dset_name)
                        if dset_name in channel_groups["EMG"]:
                            modality_to_channels["EMG"].append(dset_name)
            flag = True
            for modality, channels in modality_to_channels.items():
                if len(channels) == 0:
                    flag = False
                    break
            if flag:
                num_samples = hf[dset_name].shape[0]
                num_chunks = num_samples // chunk_size
                for chunk_start in range(0, num_chunks * chunk_size, chunk_size):
                    file_index_map.append((file_path, dset_names, chunk_start))
    except (OSError, AttributeError) as e:
        with open("problem_hdf5.txt", "a") as f:
            f.write(f"Error processing file {file_path}: {str(e)}\n")
    return file_index_map

def index_files(hdf5_paths, channel_like, samples_per_chunk, num_workers, channel_groups=None, modality_types=None):
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(pool.imap(index_file_helper, [(path, channel_like, samples_per_chunk, channel_groups, modality_types) for path in hdf5_paths]))
    return [item for sublist in results for item in sublist]


class SetTransformerDataset(Dataset):
    def __init__(self, config, channel_groups, hdf5_paths=None):
        super().__init__()
        self.config = config
        self.channel_groups = channel_groups

        # Build channel_like set
        channel_like = set()
        for mod in config["modality_types"]:
            channel_like.update(channel_groups.get(mod, []))

        if hdf5_paths:
            self.hdf5_paths = [hdf5_paths]
        else:
            data_path = Path(config["data_path"])

            # Retrieve HDF5 paths from S3 or local
            paths = list(data_path.glob("*.hdf5"))

            random.shuffle(paths)

            self.hdf5_paths = paths

        # Compute chunk size
        self.samples_per_chunk = config["sampling_duration"] * 60 * config["sampling_freq"]

        # Index files in parallel
        self.index_map = index_files(
            self.hdf5_paths,
            channel_like,
            self.samples_per_chunk,
            config["num_workers"],
            channel_groups=channel_groups,
            modality_types=config["modality_types"],
        )

        self.total_len = len(self.index_map)
        # Keep track of channels per modality
        self.modalities_length = [
            config.get(f"{mod}_CHANNELS", 0) for mod in config["modality_types"]
        ]

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        file_path, dset_names, chunk_start = self.index_map[idx]

        modality_to_channels = {modality_type: [] for modality_type in self.config["modality_types"]}
        for dset_name in dset_names:
            if dset_name in self.channel_groups["BAS"]:
                modality_to_channels["BAS"].append(dset_name)
            if dset_name in self.channel_groups["RESP"]:
                modality_to_channels["RESP"].append(dset_name)
            if dset_name in self.channel_groups["EKG"]:
                modality_to_channels["EKG"].append(dset_name)
            if dset_name in self.channel_groups["EMG"]:
                modality_to_channels["EMG"].append(dset_name)

        target = []
        with h5py.File(file_path, 'r', rdcc_nbytes=300 * 512 * 8 * 2) as hf:
            for modality_type in self.config["modality_types"]:
                num_channels = self.config[f"{modality_type}_CHANNELS"]
                data = np.zeros((len(modality_to_channels[modality_type]), self.samples_per_chunk))
                ds_names = modality_to_channels[modality_type]
                for idx, ds_name in enumerate(ds_names):
                    signal = hf[ds_name][chunk_start:chunk_start+self.samples_per_chunk]
                    data[idx] = signal
                target.append(torch.from_numpy(data).float())
        return target, file_path, dset_names, chunk_start, self.modalities_length


def collate_fn(batch):
    # Determine the number of modalities

    file_paths = [batch[i][1] for i in range(len(batch))]
    dset_names_list = [batch[i][2] for i in range(len(batch))]
    chunk_starts = [batch[i][3] for i in range(len(batch))]
    batch = [batch[i][0] for i in range(len(batch))]

    num_modalities = len(batch[0])

    # Initialize lists to hold padded data and masks for each modality
    padded_batch_list = [[] for _ in range(num_modalities)]
    mask_list = [[] for _ in range(num_modalities)]

    # Iterate over each modality
    for modality_index in range(num_modalities):
        max_channels = max(data[modality_index].shape[0] for data in batch)

        for data in batch:
            modality_data = data[modality_index]
            channels, length = modality_data.shape
            pad_channels = max_channels - channels

            # Create mask: 0 for real values, 1 for padded values
            mask = torch.cat((torch.zeros(channels), torch.ones(pad_channels)), dim=0)
            mask_list[modality_index].append(mask)

            # Pad the channel dimension
            pad_channel_tensor = torch.zeros((pad_channels, length))
            modality_data = torch.cat((modality_data, pad_channel_tensor), dim=0)

            padded_batch_list[modality_index].append(modality_data)

        # Stack the padded data and masks for the current modality
        padded_batch_list[modality_index] = torch.stack(padded_batch_list[modality_index])
        mask_list[modality_index] = torch.stack(mask_list[modality_index])

    return padded_batch_list, mask_list, file_paths, dset_names_list, chunk_starts


class CVDDataset(Dataset):
    def __init__(self,
                 config,
                 channel_groups,
                 embd_path,
                 ):

        super().__init__()

        self.config = config
        self.channel_groups = channel_groups
        self.max_channels = self.config["max_channels"]

        hdf5_paths = list(embd_path.glob("*.hdf5"))


        self.demo_data = pd.read_csv('../dataset/clinical_features.csv')

        pid_to_label = dict(zip(self.demo_data.nsrrid, self.demo_data.LABEL))


        self.index_map = [(path, pid_to_label[int(path.stem.split('-')[-1])]) for path in hdf5_paths]
        self.total_len = len(self.index_map)
        self.max_seq_len = config["model_params"]["max_seq_length"]

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        hdf5_path, label = self.index_map[idx]

        return hdf5_path, torch.tensor(label).float()

        # x_data = []
        # with h5py.File(hdf5_path, 'r') as hf:
        #     dset_names = []
        #     for dset_name in hf.keys():
        #         if isinstance(hf[dset_name], h5py.Dataset) and dset_name in self.config["modality_types"]:
        #             dset_names.append(dset_name)
        #
        #     random.shuffle(dset_names)
        #     for dataset_name in dset_names:
        #         x_data.append(hf[dataset_name][:])
        #
        # if not x_data:
        #     # Skip this data point if x_data is empty
        #     return self.__getitem__((idx + 1) % self.total_len)
        #
        # # Convert x_data list to a single numpy array
        # x_data = np.array(x_data)
        #
        # # Convert x_data to tensor
        # x_data = torch.tensor(x_data, dtype=torch.float32)
        #
        # label = torch.tensor(label, dtype=torch.float32)


        # return x_data, label, self.max_channels, self.max_seq_len, hdf5_path


def diagnosis_finetune_full_coxph_collate_fn(batch):
    x_data, label, max_channels_list, max_seq_len_list, hdf5_path_list = zip(*batch)

    num_channels = max(max_channels_list)

    if max_seq_len_list[0] == None:
        max_seq_len = max([item.size(1) for item in x_data])
    else:
        max_seq_len = max_seq_len_list[0]

    padded_x_data = []
    padded_mask = []
    for item in x_data:
        c, s, e = item.size()
        c = min(c, num_channels)
        s = min(s, max_seq_len)  # Ensure the sequence length doesn't exceed max_seq_len

        # Create a padded tensor and a mask tensor
        padded_item = torch.zeros((num_channels, max_seq_len, e))
        mask = torch.ones((num_channels, max_seq_len))

        # Copy the actual data to the padded tensor and set the mask for real data
        padded_item[:c, :s, :e] = item[:c, :s, :e]
        mask[:c, :s] = 0  # 0 for real data, 1 for padding

        padded_x_data.append(padded_item)
        padded_mask.append(mask)

    # Stack all tensors into a batch
    x_data = torch.stack(padded_x_data)
    label = torch.stack(label)
    padded_mask = torch.stack(padded_mask)

    return x_data, label, padded_mask, hdf5_path_list
if __name__ == "__main__":
    config = load_data("../configs/config_finetune_diagnosis_coxph.yaml")
    channel_groups = load_data("../configs/channel_groups.json")
    dataset = CVDDataset(config, channel_groups, Path('../checkpoints/SetTransformer/leave_one_out_128_patch_size_640/shhs_5min_agg'))

    x_data, label, max_channels, max_seq_len, hdf5_path = dataset[10]




