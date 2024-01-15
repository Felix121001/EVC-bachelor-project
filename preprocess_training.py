import os
import time
import argparse
import numpy as np
import pickle
import yaml
import shutil
import librosa
from glob import glob
import random

import preprocess


def save_pickle(variable, fileName):
    with open(fileName, "wb") as f:
        pickle.dump(variable, f)


def load_pickle_file(fileName):
    with open(fileName, "rb") as f:
        return pickle.load(f)


def load_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def split_dataset(source_dir, train_dir, test_dir, test_split):
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    wav_files = glob(os.path.join(source_dir, "*.wav"))
    random.shuffle(wav_files)  # Randomly shuffle files for unbiased split
    test_split_index = int(len(wav_files) * test_split)
    test_wav_files = wav_files[:test_split_index]
    train_wav_files = wav_files[test_split_index:]

    for wav_file in test_wav_files:
        shutil.copy(wav_file, test_dir)

    for wav_file in train_wav_files:
        shutil.copy(wav_file, train_dir)


# python preprocess_training.py


def preprocess_for_training(config_file="config.yaml", make_test_split=True, include_sp=True):
    config = load_config(config_file)
    train_dirs = config["DATASET"]["PATH"]
    cache_folder = config["DATASET"]["CACHE"]
    num_mcep = config["INPUT"]["NUM_MCEP"]
    sampling_rate = config["INPUT"]["SAMPLING_RATE"]
    frame_period = config["INPUT"]["FRAME_PERIOD"]
    run_name = config["SOLVER"]["RUN_NAME"]
    emotions = config["DATASET"]["EMOTIONS"]
    combined_dir = config["DATASET"]["TRAINING_DATASET"]

    train_dir = os.path.join(combined_dir, "train")
    test_dir = os.path.join(combined_dir, "test")

    # Create directories for combined dataset
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # cache_folder = os.path.join(cache_folder, run_name)
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    print("Starting to preprocess data.......")
    start_time = time.time()

    all_f0s_list = []
    all_coded_sps_list = []
    if include_sp: all_sp_list = []


    f0_all = {emotion: [] for emotion in emotions}
    coded_sp_all = {emotion: [] for emotion in emotions}
    if include_sp: sp_all = {emotion: [] for emotion in emotions}

    for dataset_name, dataset_path in train_dirs.items():
        for emotion in emotions:
            source_emotion_dir = os.path.join(dataset_path, emotion)
            combined_emotion_train_dir = os.path.join(train_dir, emotion)
            combined_emotion_test_dir = os.path.join(test_dir, emotion)

            if not os.path.exists(source_emotion_dir):
                print(f"Dataset {dataset_name} has no Emotion {emotion}")
                continue

            if make_test_split:
                split_dataset(
                    source_emotion_dir,
                    combined_emotion_train_dir,
                    combined_emotion_test_dir,
                    config["DATASET"]["TEST_SPLIT"],
                )

            wavs = preprocess.load_wavs(
                wav_dir=combined_emotion_train_dir, sr=sampling_rate
            )

            if not wavs:
                print(f"No WAV files found in directory: {source_emotion_dir}")
                continue
            f0s, timeaxes, sps, aps, coded_sps = preprocess.world_encode_data(
                wave=wavs,
                fs=sampling_rate,
                frame_period=frame_period,
                coded_dim=num_mcep,
                emotion=emotion,
                dataset_name=dataset_name,
            )

            transposed_coded_sps = preprocess.transpose_in_list(lst=coded_sps)
            if include_sp: transposed_sp = preprocess.transpose_in_list(lst=sps)

            f0_all[emotion].extend(f0s)
            coded_sp_all[emotion].extend(transposed_coded_sps)
            if include_sp: sp_all[emotion].extend(transposed_sp)
            all_f0s_list.extend(f0s)
            all_coded_sps_list.extend(transposed_coded_sps)
            if include_sp: all_sp_list.extend(transposed_sp)

    f0_mean, f0_std = preprocess.logf0_statistics(all_f0s_list, log=True)
    coded_sps_mean, coded_sps_std = preprocess.coded_sp_statistics(all_coded_sps_list)
    if include_sp: sp_mean, sp_std = preprocess.coded_sp_statistics(all_sp_list)

    for emotion in emotions:
        f0s_log = [np.ma.log(f0) for f0 in f0_all[emotion]]
        f0_mean_emotion, f0_std_emotion = preprocess.logf0_statistics(f0s_log, log=False)
        
        f0_all[emotion] = [(np.ma.log(f0) - f0_mean_emotion) / f0_std_emotion for f0 in f0_all[emotion]]
        
        coded_sp_all[emotion] = [
            (coded_sps_transposed - coded_sps_mean) / coded_sps_std
            for coded_sps_transposed in coded_sp_all[emotion]
        ]
        if include_sp: sp_all[emotion] = [
            (sp_transposed - sp_mean) / sp_std for sp_transposed in sp_all[emotion]
        ]

        save_pickle(
            variable=coded_sp_all[emotion],
            fileName=os.path.join(cache_folder, f"coded_sps_{emotion}_norm.pickle"),
        )
        save_pickle(
            variable=f0_all[emotion],
            fileName=os.path.join(cache_folder, f"f0_norm_{emotion}.pickle"),
        )
        
        np.savez(
            os.path.join(cache_folder, f"logf0s_{emotion}_normalization.npz"), 
            mean=f0_mean_emotion, 
            std=f0_std_emotion
        )
        
        if include_sp: save_pickle(
            variable=sp_all[emotion],
            fileName=os.path.join(cache_folder, f"sp_norm_{emotion}.pickle"),
        )

    np.savez(
        os.path.join(cache_folder, "logf0s_normalization.npz"), mean=f0_mean, std=f0_std
    )
    np.savez(
        os.path.join(cache_folder, "coded_sp_normalization.npz"),
        mean=coded_sps_mean,
        std=coded_sps_std,
    )
    if include_sp: np.savez(
        os.path.join(cache_folder, "sp_normalization.npz"),
        mean=sp_mean,
        std=sp_std,
    )

    end_time = time.time()
    print(
        "Preprocessing finished!! See your directory ../cache for cached preprocessed data"
    )
    print("Time taken for preprocessing {:.4f} seconds".format(end_time - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesses the training data for the Voice Conversion Model"
    )

    config_file = "./config.yaml"

    parser.add_argument(
        "--config_file",
        type=str,
        help="location of the config file",
        default=config_file,
    )

    argv = parser.parse_args()
    config_file = argv.config_file

    preprocess_for_training("./config.yaml", make_test_split=True)
