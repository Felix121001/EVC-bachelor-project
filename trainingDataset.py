#! python
# -*- coding: utf-8 -*-
# Author: kun
# @Time: 2019-07-23 14:36

from torch.utils.data.dataset import Dataset
import torch
import numpy as np


class trainingDataset(Dataset):
    def __init__(self, datasetA, datasetB, n_frames=128):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.n_frames = n_frames

    def __getitem__(self, index):
        dataset_A = self.datasetA
        dataset_B = self.datasetB
        n_frames = self.n_frames

        self.length = min(len(dataset_A), len(dataset_B))

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

        return train_data_A[index], train_data_B[index]

    def __len__(self):
        return min(len(self.datasetA), len(self.datasetB))


class AudioDataset(Dataset):
    def __init__(self, dataset_A, dataset_B, n_frames, f0_A=None, f0_B=None):
        self.segments_A = self.preprocess_dataset(dataset_A, f0_A, n_frames)
        self.segments_B = self.preprocess_dataset(dataset_B, f0_B, n_frames)

    def preprocess_dataset(self, dataset, f0, n_frames):
        segments = []

        if f0 is not None:
            for data, f0_it in zip(dataset, f0):
                if np.isnan(data).any():
                    raise ValueError("NaN values detected in data.")
                # if data.shape[0] != num_mcep:
                #    raise ValueError(f"Expected data to have {num_mcep} mel cepstral coefficients, but got {data.shape[0]}.")
                f0_it_expanded = np.expand_dims(f0_it, axis=0)
                # make f0 random
                # f0_it_expanded = np.random.rand(1, f0_it_expanded.shape[1])

                data = np.concatenate((data, f0_it_expanded), axis=0)
                segments.extend(self.process_data(data, n_frames))
        else:
            for data in dataset:
                if np.isnan(data).any():
                    raise ValueError("NaN values detected in data.")
                segments.extend(self.process_data(data, n_frames))
        return segments

    @staticmethod
    def process_data(data, n_frames):
        total_frames = data.shape[1]
        segments = []

        num_segments = total_frames // n_frames
        for i in range(num_segments):
            start = i * n_frames
            end = start + n_frames
            segments.append(data[:, start:end])

        leftover_frames = total_frames % n_frames
        if leftover_frames > n_frames // 4:
            padding = ((0, 0), (0, n_frames - leftover_frames))
            padded_data = np.pad(
                data[:, -leftover_frames:], padding, "constant", constant_values=(0, 0)
            )
            segments.append(padded_data)

        return segments

    def __len__(self):
        return min(len(self.segments_A), len(self.segments_B))

    def __getitem__(self, index):
        return (
            self.segments_A[index % len(self.segments_A)],
            self.segments_B[index % len(self.segments_B)],
        )


class ProcessedAudioDatasetCombined(Dataset):
    def __init__(self, datasets, f0_datasets, n_frames, one_hot_labels=False):
        self.one_hot_labels = one_hot_labels
        self.num_classes = len(datasets)

        self.dataset = []
        self.labels = []

        assert datasets.keys() == f0_datasets.keys()

        for class_idx, dataset_key in enumerate(datasets.keys()):
            dataset = datasets[dataset_key]
            f0_dataset = f0_datasets[dataset_key]

            segments = self.preprocess_dataset(dataset, f0_dataset, n_frames)
            self.dataset += segments
            self.labels += [class_idx] * len(segments)

    def preprocess_dataset(self, dataset, f0, n_frames):
        segments = []

        if f0 is not None:
            for data, f0_it in zip(dataset, f0):
                if np.isnan(data).any():
                    raise ValueError("NaN values detected in data.")
                f0_it_expanded = np.expand_dims(f0_it, axis=0)
                # make f0 random
                # f0_it_expanded = np.random.rand(1, f0_it_expanded.shape[1])

                data = np.concatenate((data, f0_it_expanded), axis=0)
                segments.extend(self.process_data(data, n_frames))
        else:
            for data in dataset:
                if np.isnan(data).any():
                    raise ValueError("NaN values detected in data.")
                segments.extend(self.process_data(data, n_frames))
        return segments

    @staticmethod
    def process_data(data, n_frames):
        total_frames = data.shape[1]
        segments = []

        num_segments = total_frames // n_frames
        for i in range(num_segments):
            start = i * n_frames
            end = start + n_frames
            segments.append(data[:, start:end])

        leftover_frames = total_frames % n_frames
        if leftover_frames > n_frames // 2 or (leftover_frames == total_frames):
            padding = ((0, 0), (0, n_frames - leftover_frames))
            padded_data = np.pad(
                data[:, -leftover_frames:], padding, "constant", constant_values=(0, 0)
            )
            segments.append(padded_data)

        return segments

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        label = self.labels[index]

        if self.one_hot_labels:
            one_hot_label = torch.zeros(self.num_classes)
            one_hot_label[label] = 1
            return torch.from_numpy(item).float(), one_hot_label
        else:
            return torch.from_numpy(item).float(), torch.tensor(
                label, dtype=torch.float32
            )
