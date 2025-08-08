import os
import glob
import numpy as np
import h5py
import warnings
import multiprocessing
import argparse
from pathlib import Path
from loguru import logger
from scipy.signal import butter, filtfilt
import mne
from tqdm import tqdm
from itertools import chain
import shutil
import warnings

warnings.filterwarnings('ignore')


class EDFToHDF5Converter:
    def __init__(
        self,
        root_dir: str,
        target_dir: str,
        resample_rate: int = 512,
        num_threads: int = 1,
        num_files: int = -1
    ):
        self.resample_rate = resample_rate
        self.root_dir = Path(root_dir.rstrip('/'))
        self.target_dir = Path(target_dir.rstrip('/'))
        self.num_threads = num_threads
        self.num_files = num_files

        self.unique_ch = set()

        # file list
        self.file_paths, self.file_names = self.get_files()



    def get_files(self):

        file_paths = list(self.root_dir.glob('*'))
        file_names = [path.name for path in file_paths]

        return file_paths, file_names

    def read_edf(self, file_path):

        logger.info('reading edf')
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        signals = [raw.get_data(picks=[ch_name])[0] for ch_name in raw.ch_names]
        sample_rates = np.array([raw.info['sfreq'] for _ in raw.ch_names])
        channel_names = np.array(raw.ch_names)

        return signals, sample_rates, channel_names

    def safe_standardize(self, signal: np.ndarray) -> np.ndarray:
        mean = np.mean(signal)
        std = np.std(signal)

        return (signal - mean) / std if std != 0 else (signal - mean)

    def filter_signal(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        nyquist = sample_rate / 2
        cutoff = min(self.resample_rate / 2, nyquist)
        b, a = butter(4, cutoff/nyquist, btype='low')
        return filtfilt(b, a, signal)

    def resample_signals(self, signals, sample_rates):
        resampled = []
        for sig, rate in zip(signals, sample_rates):
            duration = len(sig) / rate
            orig_t = np.linspace(0, duration, num=len(sig), endpoint=False)
            new_count = int(duration * self.resample_rate)
            new_t = np.linspace(0, duration, num=new_count, endpoint=False)
            if rate > self.resample_rate:
                sig = self.filter_signal(sig, rate)
            resig = np.interp(new_t, orig_t, sig)
            stdsig = self.safe_standardize(resig)

            if np.isnan(stdsig).any():
                logger.warning("NaNs in resampled signal; skipping channel")
                continue
            resampled.append(stdsig)
        return np.stack(resampled)

    def _get_unique_name(self, hdf, base_name):
        name = base_name
        i = 1
        while name in hdf:
            name = f"{base_name}_{i}"
            i += 1
        return name

    def save_to_hdf5(self, signals, channel_names, file_path):

        logger.info('saving hdf5')
        samples_per_chunk = 5 * 60 * self.resample_rate
        with h5py.File(file_path, 'w') as hdf:
            for signal, name in zip(signals, channel_names):
                dataset_name = self._get_unique_name(hdf, name)
                hdf.create_dataset(dataset_name, data=signal,
                                   dtype='float16', chunks=(samples_per_chunk,), compression="gzip")


    def convert(self, edf_path, hdf5_path):
        signals, rates, names = self.read_edf(edf_path)
        resigs = self.resample_signals(signals, rates)
        edf_path.unlink()
        self.save_to_hdf5(resigs, names, hdf5_path)

        return names

    def convert_multiprocessing(self, edf_list):

        chunk_ch = set()

        for edf_file in tqdm(edf_list, total=len(edf_list), desc="Converting EDFs"):
            stem = edf_file.stem
            h5_name = f"{stem}.hdf5"

            h5_path = self.target_dir / h5_name

            if h5_path.exists():
                logger.info(f"Skipping existing: {h5_path}")
                continue

            try:
                ch_names = self.convert(edf_file, h5_path)
                chunk_ch.update(ch_names)
            except Exception as e:
                warnings.warn(f"Could not process {edf_file}: {e}")
        return chunk_ch

    def convert_all_multiprocessing(self):
        files = self.file_paths
        if self.num_files != -1:
            files = files[:self.num_files]
        chunks = np.array_split(files, self.num_threads)
        with multiprocessing.Pool(self.num_threads) as pool:
            results = pool.map(self.convert_multiprocessing, chunks)

        for worker_set in results:
            self.unique_ch.update(worker_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default="/temp_work/ch266186/selected_patients/shhs2")
    parser.add_argument('--target_dir', default="/temp_work/ch266186/shhs_hdf5")
    parser.add_argument('--num_threads', type=int, default=8)
    parser.add_argument('--num_files', type=int, default=-1)
    parser.add_argument('--resample_rate', type=int, default=128)
    args = parser.parse_args()

    converter = EDFToHDF5Converter(
        root_dir=args.root_dir,
        target_dir=args.target_dir,
        num_threads=args.num_threads,
        num_files=args.num_files,
        resample_rate=args.resample_rate
    )
    converter.convert_all_multiprocessing()

    unique_channels = sorted(converter.unique_ch)

    with open("ch_names.txt", "w") as f:
        for ch in unique_channels:
            f.write(f"{ch}\n")
