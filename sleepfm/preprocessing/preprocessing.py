
import os
import glob
import pyedflib
import h5py
import numpy as np
import pandas as pd
import datetime
from scipy.signal import resample
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from loguru import logger
import argparse
import warnings
from scipy.signal import butter, filtfilt
import mne
import s3fs
from itertools import chain
from pathlib import Path



class EDFToHDF5Converter:
    def __init__(self, root_dir, target_dir, resample_rate=512, num_threads=1, num_files=-1):
        self.resample_rate = resample_rate 
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.num_threads = num_threads
        self.num_files = num_files
        self.file_locations = self.get_files()


    def get_files(self):
        fs = s3fs.S3FileSystem(anon=False)

        # 1. List the IDs you care about
        subject_ids = ["I0004", "I0006"]

        # 2. Build a list of glob results, one per subject
        all_lists = [
            fs.glob(
                f"arn:aws:s3:us-east-1:184438910517:accesspoint/"
                f"bdsp-psg-access-point/PSG/bids/{sid}/**/eeg/*.[eE][dD][fF]"
            )
            for sid in tqdm(subject_ids, total=2)
        ]

        # 3. Flatten into a single list of keys
        edf_keys = list(chain.from_iterable(all_lists))
        file_paths = [Path(key) for key in edf_keys]
        file_names = [path.name for path in file_paths]

        return file_paths, file_names
    
    def extract_start_time(self, file_path):
        with open(file_path, 'r') as file:
            lines = [next(file) for _ in range(5)]
            
        for line in lines:
            if line.startswith("Start Time:"):
                return line.split(": ", 1)[1].strip()
        return "Start Time not found"


    def read_edf(self, file_path):

        logger.info('reading edf')
        raw = mne.io.read_raw_edf(file_path, preload=True)
        signals = [raw.get_data(picks=[ch_name])[0] for ch_name in raw.ch_names]
        sample_rates = np.array([raw.info['sfreq'] for _ in raw.ch_names])
        channel_names = np.array(raw.ch_names)

        return signals, sample_rates, channel_names


    def safe_standardize(self, signal):
        mean = np.mean(signal)
        std = np.std(signal)
        
        if std == 0:
            standardized_signal = (signal - mean)
        else:
            standardized_signal = (signal - mean) / std
        
        return standardized_signal
        
    def filter_signal(self, signal, sample_rate):
        print("Filtering signal")
        nyquist_freq = sample_rate / 2
        cutoff = min(self.resample_rate / 2, nyquist_freq)
        normalized_cutoff = cutoff / nyquist_freq
        b, a = butter(4, normalized_cutoff, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    def resample_signals(self, signals, sample_rates):
        logger.info('resampling signals')
        resampled_signals = []
        for signal, rate in zip(signals, sample_rates):
            # Calculate the duration of the signal
            duration = len(signal) / rate
            
            # Original time points
            original_time_points = np.linspace(0, duration, num=len(signal), endpoint=False)
            
            # New sample rate and new time points
            new_sample_count = int(duration * self.resample_rate)
            new_time_points = np.linspace(0, duration, num=new_sample_count, endpoint=False)

            #filter signal
            if rate > self.resample_rate:
                signal = self.filter_signal(signal, rate)
            
            # Linear interpolation
            resampled_signal = np.interp(new_time_points, original_time_points, signal)

            standardized_signal = self.safe_standardize(resampled_signal)
            
            if np.isnan(standardized_signal).any():
                logger.info('Found NaN in the resampled signal.')
                # Handle the NaN case here (e.g., skip or fix the signal)
                continue

            resampled_signals.append(standardized_signal)

        
        return np.stack(resampled_signals)

    def save_to_hdf5(self, signals, channel_names, file_path):
        logger.info('saving hdf5')
        samples_per_chunk = 5 * 60 * self.resample_rate
        with h5py.File(file_path, 'w') as hdf:
            for signal, name in zip(signals, channel_names):
                dataset_name = self._get_unique_name(hdf, name)
                hdf.create_dataset(dataset_name, data=signal,
                                   dtype='float16', chunks=(samples_per_chunk,), compression="gzip")


    def _get_unique_name(self, hdf, base_name):
        # Helper method to ensure dataset names are unique
        i = 1
        unique_name = base_name
        while unique_name in hdf:
            unique_name = f"{base_name}_{i}"
            i += 1
        return unique_name

    
    def convert(self, edf_path, hdf5_path):
        signals, sample_rates, channel_names = self.read_edf(edf_path)
        resampled_signals = self.resample_signals(signals, sample_rates)
        self.save_to_hdf5(resampled_signals, channel_names, hdf5_path)

    def convert_multiprocessing(self, args):
        edf_files = args

        for edf_file in tqdm(edf_files, desc="Converting EDF files"):

            if edf_file.endswith(".edf"):
                replace_str = ".edf"
            elif edf_file.endswith(".EDF"):
                replace_str = ".EDF"
            hdf5_file = os.path.join(self.target_dir, edf_file.split('/')[-1].replace(replace_str, '.hdf5'))

            if os.path.exists(hdf5_file):
                logger.info(f"File already processed: {hdf5_file}")
                continue
            try:
                self.convert(edf_file, hdf5_file)
            except Exception as e:
                warnings.warn(f"Warning: Could not process the file {edf_file}. Error: {str(e)}")
                continue
        return [1]

    def convert_all(self):
        edf_files, edf_names = self.get_files() 
        # folders = self.get_folders()
        for edf_file in tqdm(edf_files, desc="Converting EDF files"):

            if edf_file.endswith(".edf"):
                replace_str = ".edf"
            elif edf_file.endswith(".EDF"):
                replace_str = ".EDF"
            hdf5_file = os.path.join(self.target_dir, edf_file.split('/')[-1].replace(replace_str, '.hdf5'))


            try:
                self.convert(edf_file, hdf5_file)
            except Exception as e:
                warnings.warn(f"Warning: Could not process the file {edf_file}. Error: {str(e)}")
                continue

    def convert_all_multiprocessing(self):
        edf_files, edf_names = self.get_files() 

        if self.num_files != -1:
            edf_files = edf_files[:self.num_files]

        edf_files_chunks = np.array_split(edf_files, self.num_threads)
        tasks = [(edf_files_chunk) for edf_files_chunk in edf_files_chunks]
        with multiprocessing.Pool(self.num_threads) as pool:
            preprocessed_results = [y for x in pool.imap_unordered(self.convert_multiprocessing, tasks) for y in x]



    def plot_results(self, resampled_signals, channel_names):
        print("plotting resampled_signals")
        num_signals = len(resampled_signals)
        fig, axs = plt.subplots(num_signals, 1, figsize=(15, 3*num_signals), sharex=True)
        samples_to_plot = 10 * self.resample_rate
        sample_to_start = 10 * self.resample_rate
        for i, (signal, name) in enumerate(zip(resampled_signals, channel_names)):
            signal_chunk = signal[sample_to_start:sample_to_start+samples_to_plot]
            axs[i].plot(signal_chunk)
            axs[i].set_title(name)
            axs[i].set_ylabel('Amplitude')
        
        axs[-1].set_xlabel('Samples')
        plt.tight_layout()
        plt.show()

    def plot_first_results(self, resampled_signals, channel_names):
        print("plotting resampled_signals")
        num_signals = len(resampled_signals)
        fig = plt.figure(figsize=(15, 3))
        samples_to_plot = 10 * self.resample_rate
        sample_to_start = 10 * self.resample_rate
        for i, (signal, name) in enumerate(zip(resampled_signals, channel_names)):
            signal_chunk = signal[sample_to_start:sample_to_start+samples_to_plot]
            plt.plot(signal_chunk)
            plt.title(name)
            plt.ylabel('Amplitude')
            break
        
        plt.xlabel('Samples')
        plt.tight_layout()
        plt.show()

    def process_and_plot_single_file(self, edf_path):
        signals, sample_rates, channel_names = self.read_edf(edf_path)
        resampled_signals = self.resample_signals(signals, sample_rates)

        self.plot_first_results(resampled_signals, channel_names)
    

def main():
    parser = argparse.ArgumentParser(description="Process data and create hdf5")
    parser.add_argument('--root_dir', type=str, required=True, help='Path to edf')
    parser.add_argument('--target_dir', type=str, required=True, help='Path to save hdf5')
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for parallel processing")
    parser.add_argument("--num_files", type=int, default=-1, help="Number of files to process. If -1, process all")
    parser.add_argument("--resample_rate", type=int, default=256, help="Target sampling rate for hdf5 file.")
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    converter = EDFToHDF5Converter(root_dir=args.root_dir,
                                target_dir=args.target_dir, 
                                num_threads=args.num_threads, 
                                num_files=args.num_files,
                                resample_rate=args.resample_rate)

    converter.convert_all_multiprocessing()


if __name__ == "__main__":
    main()



