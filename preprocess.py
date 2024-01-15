#! python
# -*- coding: utf-8 -*-
# Author: kun
# @Time: 2019-07-23 14:26


import librosa
import numpy as np
import os
import pyworld
from pprint import pprint
import librosa.display
import time


##################### added by felix to speed up ############################
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import multiprocessing


def load_wav(file_path, sr):
    wav, _ = librosa.load(file_path, sr=sr, mono=True)
    return wav


def world_encode_data_single(wav, fs, frame_period, coded_dim):
    f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=fs, frame_period=frame_period)
    coded_sp = world_encode_spectral_envelop(sp=sp, fs=fs, dim=coded_dim)
    return f0, timeaxis, sp, ap, coded_sp


def load_wavs(wav_dir, sr=16000):
    print("-- load wavs {}...".format(wav_dir))
    files = [f for f in sorted(os.listdir(wav_dir)) if f.endswith(".wav")]
    file_paths = [os.path.join(wav_dir, f) for f in files]

    pool = Pool(processes=os.cpu_count())
    load_wav_partial = partial(load_wav, sr=sr)
    wavs = list(tqdm(pool.imap(load_wav_partial, file_paths), total=len(file_paths)))

    pool.close()
    pool.join()

    return wavs


def world_encode_data(
    wave, fs, frame_period=5.0, coded_dim=24, emotion="None", dataset_name="None"
):
    print("-- world_encode_data {} {}...".format(emotion, dataset_name))

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)

    encode_func_partial = partial(
        world_encode_data_single, fs=fs, frame_period=frame_period, coded_dim=coded_dim
    )

    results = list(tqdm(pool.imap(encode_func_partial, wave), total=len(wave)))

    pool.close()
    pool.join()

    f0s, timeaxes, sps, aps, coded_sps = zip(*results)

    return list(f0s), list(timeaxes), list(sps), list(aps), list(coded_sps)




def world_decompose(wav, fs, frame_period=5.0):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(
        wav, fs, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0
    )

    # Finding Spectogram
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)

    # Finding aperiodicity
    ap = pyworld.d4c(wav, f0, timeaxis, fs)

    # Use this in Ipython to see plot
    # librosa.display.specshow(np.log(sp).T,
    #                          sr=fs,
    #                          hop_length=int(0.001 * fs * frame_period),
    #                          x_axis="time",
    #                          y_axis="linear",
    #                          cmap="magma")
    # colorbar()
    return f0, timeaxis, sp, ap


def world_encode_spectral_envelop(sp, fs, dim=24):
    # Get Mel-Cepstral coefficients (MCEPs)
    # sp = sp.astype(np.float64)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)
    return coded_sp


def logf0_statistics(f0s, log=True):
    # Note: np.ma.log() calculating log on masked array (for incomplete or invalid entries in array)
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()
    return log_f0s_mean, log_f0s_std


def coded_sp_statistics(coded_sps):
    coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=True)
    return coded_sps_mean, coded_sps_std


def logf0_normalization(f0s_A, f0s_B, log=False):
    print(len(f0s_A))
    concatenated = np.ma.log(np.concatenate(f0s_A + f0s_B))
    f0_mean = concatenated.mean()
    f0_std = concatenated.std()
    if log:
        func = lambda x: np.log(x + 1e-10)
    else:
        func = lambda x: x
    normalized_A = list()
    for f0 in f0s_A:
        normalized_A.append((func(f0) - f0_mean) / f0_std)
    normalized_B = list()
    for f0 in f0s_B:
        normalized_B.append((func(f0) - f0_mean) / f0_std)
    return normalized_A, normalized_B, f0_mean, f0_std


def transpose_in_list(lst):
    transposed_lst = list()
    for array in lst:
        transposed_lst.append(array.T)
    return transposed_lst


def coded_sps_normalization_fit_transform(coded_sps):
    coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)
    return coded_sps_normalized, coded_sps_mean, coded_sps_std


def coded_sps_normalization_fit_transform_comb(coded_sps_A, coded_sps_B):
    coded_sps_concatenated = np.concatenate(coded_sps_A + coded_sps_B, axis=1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_normalized_A = list()
    for coded_sp in coded_sps_A:
        coded_sps_normalized_A.append((coded_sp - coded_sps_mean) / coded_sps_std)
    coded_sps_normalized_B = list()
    for coded_sp in coded_sps_B:
        coded_sps_normalized_B.append((coded_sp - coded_sps_mean) / coded_sps_std)
    return coded_sps_normalized_A, coded_sps_normalized_B, coded_sps_mean, coded_sps_std


def wav_padding(wav, sr, frame_period, multiple=4):
    assert wav.ndim == 1
    num_frames = len(wav)
    num_frames_padded = int(
        (
            np.ceil(
                (np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1
            )
            * multiple
            - 1
        )
        * (sr * frame_period / 1000)
    )
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(
        wav, (num_pad_left, num_pad_right), "constant", constant_values=0
    )

    return wav_padded


def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):
    # Logarithm Gaussian Normalization for Pitch Conversions
    f0_converted = np.exp(
        (np.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target
    )
    return f0_converted


def world_decode_spectral_envelop(coded_sp, fs):
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)
    return decoded_sp


def world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period):
    f0_double = f0.astype(np.float64)
    decoded_sp_double = decoded_sp.astype(np.float64)
    ap_double = ap.astype(np.float64)

    wav = pyworld.synthesize(f0_double, decoded_sp_double, ap_double, fs, frame_period)
    wav = wav.astype(np.float32)
    return wav


def check_data(f0, decoded_sp, ap):
    data_dict = {"f0": f0, "decoded_sp": decoded_sp, "ap": ap}

    for name, data in data_dict.items():
        # Check for NaN values
        if np.isnan(data).any():
            print(f"{name} contains NaN values")

        # Check for infinite values
        if np.isinf(data).any():
            print(f"{name} contains infinite values")

        # Check for very large values
        max_val = np.finfo(data.dtype).max
        if np.abs(data).max() > max_val / 10:  # Adjust threshold as needed
            print(f"{name} contains very large values")

        # Check data shape
        print(f"{name} shape: {data.shape}")


def sample_train_data(dataset_A, dataset_B, n_frames=128):
    # Created Pytorch custom dataset instead
    num_samples = min(len(dataset_A), len(dataset_B))
    train_data_A_idx = np.arange(len(dataset_A))
    train_data_B_idx = np.arange(len(dataset_B))
    np.random.shuffle(train_data_A_idx)
    np.random.shuffle(train_data_B_idx)
    train_data_A_idx_subset = train_data_A_idx[:num_samples]
    train_data_B_idx_subset = train_data_B_idx[:num_samples]

    train_data_A = list()
    train_data_B = list()

    for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
        data_A = dataset_A[idx_A]
        frames_A_total = data_A.shape[1]
        assert frames_A_total >= n_frames
        start_A = np.random.randint(frames_A_total - n_frames + 1)
        end_A = start_A + n_frames
        train_data_A.append(data_A[:, start_A:end_A])

        data_B = dataset_B[idx_B]
        frames_B_total = data_B.shape[1]
        assert frames_B_total >= n_frames
        start_B = np.random.randint(frames_B_total - n_frames + 1)
        end_B = start_B + n_frames
        train_data_B.append(data_B[:, start_B:end_B])

    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)

    return train_data_A, train_data_B

