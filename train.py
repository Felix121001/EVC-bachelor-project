#! python
# -*- coding: utf-8 -*-
# Author: kun
# @Time: 2019-07-23 14:25

import os
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import time
import librosa
import pickle
import soundfile as sf
import yaml

import preprocess
from trainingDataset import trainingDataset, AudioDataset, ProcessedAudioDatasetCombined
from GAN.discriminator_model import Discriminator
from GAN.generator_model import (
    build_unet,
    build_unet_residual,
    DualPathwayModel,
    SeperatePathwayGenerator,
    CombinedTransformerGenerator,
)
from GAN.classifier import Classifier
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"









def smooth_f0(f0, window_size=5):
    """
    Smooths the F0 contour using a moving average filter.
    
    Parameters:
    f0 (numpy array): The F0 contour array.
    window_size (int): The size of the moving average window. Must be an odd integer.
    
    Returns:
    numpy array: The smoothed F0 contour.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd integer")

    half_window = window_size // 2
    f0_padded = np.pad(f0, (half_window, half_window), mode='reflect')
    f0_smooth = np.convolve(f0_padded, np.ones(window_size) / window_size, mode='valid')

    return f0_smooth




class TrackLosses:
    def __init__(self, loss_names):
        self.loss_names = loss_names
        self.losses = {loss_name: [] for loss_name in loss_names}

    def __getitem__(self, loss_name):
        return self.losses[loss_name]

    def update(self, loss_values):
        for loss_name, loss_value in zip(self.loss_names, loss_values):
            self.losses[loss_name].append(loss_value)

    def reset(self):
        for loss_name in self.loss_names:
            self.losses[loss_name] = []

    def get_last_average(self, loss_name, n=50):
        return np.mean(self.losses[loss_name][-n:])

    def plot(self, save_path=None):
        for loss_name in self.loss_names:
            plt.plot(self.losses[loss_name], label=loss_name)
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


class CustomLRScheduler:
    def __init__(self, optimizer, base_lr, decay_factor, num_steps, min_lr):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.decay_factor = decay_factor
        self.num_steps = num_steps
        self.current_step = 0
        self.min_lr = min_lr

    def step(self):
        self.current_step += 1
        lr = self.base_lr * (self.decay_factor ** (self.current_step / self.num_steps))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = max(
                lr, self.min_lr
            )  # Ensure lr doesn't go below a threshold

    def reset(self):
        self.current_step = 0
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.base_lr



class CycleGANTraining(object):
    def __init__(self, config_file, restart_training_at=None):
        self.configs = self.load_config(config_file)

        self.start_epoch = 0
        self.num_epochs = self.configs["SOLVER"]["MAX_EPOCHS"]
        self.mini_batch_size = self.configs["SOLVER"]["BATCH_SIZE"]
        self.cycle_loss_lambda = self.configs["SOLVER"]["CYCLE_LOSS_LAMBDA"]
        self.identity_loss_lambda = self.configs["SOLVER"]["IDENTITY_LOSS_LAMBDA"]
        self.emotion_loss_lambda = self.configs["SOLVER"]["EMOTION_LOSS_LAMBDA"]
        self.feature_loss_lambda = self.configs["SOLVER"]["FEATURE_MATCHING_LOSS_LAMBDA"]
        self.tv_loss_lambda = self.configs["SOLVER"]["TV_LOSS_LAMBDA"]
        self.use_f0_model = self.configs["SOLVER"]["USE_F0_MODEL"]

        self.n_frames = self.configs["INPUT"]["N_FRAMES"]
        self.num_mcep = self.configs["INPUT"]["NUM_MCEP"]
        self.sampling_rate = self.configs["INPUT"]["SAMPLING_RATE"]
        self.frame_period = self.configs["INPUT"]["FRAME_PERIOD"]

        self.current_run_name = self.configs["SOLVER"]["RUN_NAME"]
        self.cache_dir = self.configs["DATASET"]["CACHE"]
        # self.cache_dir = os.path.join(self.cache_dir, self.current_run_name)

        self.emotions = self.configs["DATASET"]["EMOTIONS"]
        self.num_emotions = len(self.emotions)
        self.sp_datasets, self.f0_datasets = self.load_emotion_data(
            self.cache_dir, self.emotions
        )
        self.logf0_stats, self.sps_stats, self.logf0_stats_emo = self.load_statistics(self.cache_dir)
        self.test_dir = os.path.join(
            self.configs["DATASET"]["TRAINING_DATASET"], "test"
        )

        self.combined_sp_norm = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.architecture_G = self.configs["MODEL"]["ARCHITECTURE_G"]
        self.architecture_D = self.configs["MODEL"]["ARCHITECTURE_D"]
        
        match self.architecture_G:
            case "SeperatePathwayGenerator":
                self.generator = SeperatePathwayGenerator(self.configs).to(self.device)
            case "CombinedTransformerGenerator":
                self.generator = CombinedTransformerGenerator(self.configs).to(self.device)
            case "DualPathwayModel":
                self.generator = DualPathwayModel(self.configs).to(self.device)
            case "build_unet":
                self.generator = build_unet(self.configs).to(self.device)
            case _:
                raise ValueError("Generator architecture not supported")

        match self.architecture_D:
            case "Discriminator":
                self.discriminator = Discriminator(self.configs).to(self.device)
        
        self.classifier = Classifier(self.configs).to(self.device)


        self.scaler = GradScaler()
        g_params = self.generator.parameters()
        d_params = self.discriminator.parameters()

        self.loss_tracker = TrackLosses(
            [
                "g_loss",
                "d_loss",
                "g_cycle_loss",
                "g_id_loss",
                "g_emotion_loss",
                "g_feature_matching_loss",
                "g_tv_loss",
            ]
        )
        self.test_loss_tracker = TrackLosses(
            [
                "g_loss",
                "d_loss",
                "g_cycle_loss",
                "g_id_loss",
                "g_emotion_loss",
                "g_feature_matching_loss",
                "g_tv_loss",
            ]
        )

        self.generator_lr = self.configs["SOLVER"]["LR_G"]
        self.discriminator_lr = self.configs["SOLVER"]["LR_D"]

        self.generator_lr_decay = (
            self.generator_lr / self.configs["SOLVER"]["LR_DECAY_ITER"]
        )
        self.discriminator_lr_decay = (
            self.discriminator_lr / self.configs["SOLVER"]["LR_DECAY_ITER"]
        )
        self.start_decay = self.configs["SOLVER"]["START_LR_DECAY"]
        self.min_lr = self.configs["SOLVER"]["MIN_LR"]

        if self.configs["SOLVER"]["OPTIMIZER"] == "adam":
            self.generator_optimizer = torch.optim.Adam(
                g_params, lr=self.generator_lr, betas=(0.5, 0.999)
            )
            self.discriminator_optimizer = torch.optim.Adam(
                d_params, lr=self.discriminator_lr, betas=(0.5, 0.999)
            )
        else:
            print("Optimizer not supported")

        self.modelCheckpoint = self.configs["SOLVER"]["CHECKPOINT_DIR"]
        os.makedirs(self.modelCheckpoint, exist_ok=True)
        self.modelCheckpoint = os.path.join("model_checkpoint", self.current_run_name)
        os.makedirs(self.modelCheckpoint, exist_ok=True)

        os.makedirs("converted_sound", exist_ok=True)
        self.base_conversion_dir = os.path.join(
            "converted_sound", self.current_run_name
        )
        os.makedirs(self.base_conversion_dir, exist_ok=True)

        self.file_name = "log_store_non_sigmoid.txt"

        run_directory = os.path.join("runs", self.current_run_name)
        validation_directory = os.path.join(run_directory, "validation")
        train_directory = os.path.join(run_directory, "train")
        os.makedirs(run_directory, exist_ok=True)
        os.makedirs(validation_directory, exist_ok=True)
        os.makedirs(train_directory, exist_ok=True)
        self.writer_train = SummaryWriter(train_directory)
        self.writer_validation = SummaryWriter(validation_directory)

        if restart_training_at is not None:
            # Training will resume from previous checkpoint
            # create path if it does not exist
            self.start_epoch = self.loadModel(restart_training_at)
            print("Training resumed")

    def adjust_lr_rate(self, optimizer, name="generator"):
        if name == "generator":
            self.generator_lr = max(0.0001, self.generator_lr - self.generator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups["lr"] = self.generator_lr
        else:
            self.discriminator_lr = max(
                0.00005, self.discriminator_lr - self.discriminator_lr_decay
            )
            for param_groups in optimizer.param_groups:
                param_groups["lr"] = self.discriminator_lr

    def reset_grad(self):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    def train(self):
        print(self.device)

        dataset = ProcessedAudioDatasetCombined(
            datasets=self.sp_datasets,
            f0_datasets=self.f0_datasets,
            n_frames=self.n_frames,
            one_hot_labels=True,
        )

        n_samples = len(dataset)
        val_split = self.configs["DATASET"]["VAL_SPLIT"]
        n_val_samples = int(n_samples * val_split)
        print("Number of training samples: {}".format(n_samples))
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [n_samples - n_val_samples, n_val_samples]
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.mini_batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.mini_batch_size,
            shuffle=True,
            drop_last=False,
        )

        if self.configs["SOLVER"]["TRAIN_CLASSIFIER"] == True:
            self.train_classifier(train_loader)
        else:
            self.load_classifier()

        if True:
            # testing Set
            print("Generating testing... ,epoch ", self.start_epoch - 1)
            testing_start_time = time.time()

            #self.testing(self.generator, "angry", "neutral", self.start_epoch - 1, 0)
            #self.testing(self.generator, "neutral", "angry", self.start_epoch - 1, 0)
            #self.testing(self.generator, "neutral", "happy", self.start_epoch - 1, 0)
            #self.testing(self.generator, "neutral", "sad", self.start_epoch - 1, 0)

            testing_end_time = time.time()

            store_to_file = "Time taken for testing Set: {}".format(
                testing_end_time - testing_start_time
            )
            self.store_to_file(store_to_file)
            print(
                "Time taken for testing Set: {}".format(
                    testing_end_time - testing_start_time
                )
            )
        """""
        generator_scheduler = CustomLRScheduler(
            self.generator_optimizer,
            base_lr=self.generator_lr,  
            decay_factor=0.01, 
            num_steps=len(train_loader), 
            min_lr=self.min_lr, 
        )
        """""

        for epoch in range(self.start_epoch, self.num_epochs):
            start_time_epoch = time.time()
            self.loss_tracker.reset()
            #generator_scheduler.reset()

            for i, (real, class_enc) in enumerate(train_loader):
                num_iterations = (n_samples // self.mini_batch_size) * epoch + i

                # if num_iterations > self.start_decay:
                #    self.adjust_lr_rate(self.generator_optimizer, name='generator')
                #    self.adjust_lr_rate(self.discriminator_optimizer, name='discriminator')
                #generator_scheduler.step()

                real = real.to(self.device).float()
                class_enc = class_enc.to(self.device)  # torch.Size([16, 4])

                batch_size = class_enc.size(0)
                num_emotions = self.num_emotions

                random_class_enc = torch.empty(
                    batch_size, dtype=torch.long, device=self.device
                )

                for i in range(batch_size):
                    while True:
                        random_emotion = torch.randint(
                            0, num_emotions, (1,), device=self.device
                        )
                        if class_enc[i].argmax() != random_emotion:
                            break
                    random_class_enc[i] = random_emotion
                # random_class_enc = torch.randint(0, self.num_emotions, (class_enc.size(0),), device=self.device) #torch.Size([16])
                random_class_enc = F.one_hot(
                    random_class_enc, self.num_emotions
                ).float()  # torch.Size([16, 4])

                fake = self.generator(real, random_class_enc)

                cycle = self.generator(fake, class_enc)

                identity = self.generator(real, class_enc)

                d_fake_real_fake, fake_features = self.discriminator(fake)
                _, real_features = self.discriminator(real)

                feature_matching_loss = F.l1_loss(fake_features, real_features.detach())

                cycleLoss = torch.mean(torch.abs(real - cycle))

                identiyLoss = torch.mean(torch.abs(real - identity))

                generator_loss = torch.mean((1 - d_fake_real_fake) ** 2)

                # tv_loss = (self.total_variation_loss_spectrogram(fake) + self.total_variation_loss_spectrogram(cycle) + self.total_variation_loss_spectrogram(identity)) / 3

                predicted_emotion = self.classifier(fake)

                generator_emotion_loss = F.cross_entropy(
                    predicted_emotion, random_class_enc.argmax(dim=1)
                )

                g_loss = (
                    generator_loss
                    + self.cycle_loss_lambda * cycleLoss
                    + self.identity_loss_lambda * identiyLoss
                    + self.emotion_loss_lambda * generator_emotion_loss
                )

                if True:
                    g_loss += self.feature_loss_lambda * feature_matching_loss
                    # + tv_loss * self.tv_loss_lambda
                    

                self.reset_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(), max_norm=2.0
                )
                self.generator_optimizer.step()

                noise = torch.randn_like(real) * 0.05  
                noisy_real = real + noise

                d_real_real_fake, _ = self.discriminator(noisy_real)
                d_fake_real_fake, _ = self.discriminator(
                    fake.detach()
                )  # detach to avoid backprop through generator

                d_loss_real = torch.mean((1 - d_real_real_fake) ** 2)
                d_loss_fake = torch.mean((0 - d_fake_real_fake) ** 2)
                d_loss = (d_loss_real + d_loss_fake) / 2.0

                self.reset_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), max_norm=1.0
                )
                self.discriminator_optimizer.step()

                feature_matching_loss = 0
                tv_loss = 0

                self.loss_tracker.update(
                    [
                        g_loss.item(),
                        d_loss.item(),
                        cycleLoss.item(),
                        identiyLoss.item(),
                        generator_emotion_loss.item(),
                        0,
                        0,
                        d_loss.item(),
                    ]
                )

                if (i + 1) % self.configs["SOLVER"]["LOG_INTERVAL"] == 0:
                    description = "Epoch:{} Iter:{} lr_G:{} lr_D:{} G_Loss:{:.4f} D_Loss:{:.4f} G_id:{:.4f} G_cyc:{:.4f} G_em:{:.4f} G_fe:{:.4f} G_tv:{:.4f}".format(
                        epoch,
                        num_iterations,
                        self.generator_optimizer.param_groups[0]["lr"],
                        self.discriminator_optimizer.param_groups[0]["lr"],
                        g_loss.item(),
                        d_loss.item(),
                        identiyLoss.item(),
                        cycleLoss.item(),
                        generator_emotion_loss.item(),
                        0,
                        0,
                    )
                    print(description, end="\r", flush=True)

                if num_iterations % self.configs["SOLVER"]["LOG_TS_BOARD_INTERVAL"] == 0:
                    train_metrics = {
                        "losses/av_d_loss": "d_loss",
                        "losses/av_g_loss": "g_loss",
                        "losses/av_g_cycle_loss": "g_cycle_loss",
                        "losses/av_g_identity_loss": "g_id_loss",
                        "losses/av_g_emotion_loss": "g_emotion_loss",
                        "losses/av_g_feature_matching_loss": "g_feature_matching_loss",
                        "losses/av_g_tv_loss": "g_tv_loss"
                    }

                    for metric_name, loss_key in train_metrics.items():
                        average_loss = self.loss_tracker.get_last_average(loss_key)
                        self.writer_train.add_scalar(metric_name, average_loss, num_iterations)

                    val_losses = self.calculate_val_error(val_loader)
                    val_metrics = {
                        "losses_val/av_d_loss": val_losses[1],
                        "losses_val/av_g_loss": val_losses[0],
                        "losses_val/av_g_cycle_loss": val_losses[2],
                        "losses_val/av_g_identity_loss": val_losses[3],
                        "losses_val/av_g_emotion_loss": val_losses[4],
                        "losses_val/av_g_feature_matching_loss": val_losses[5],
                        "losses_val/av_g_tv_loss": val_losses[6]
                    }

                    for metric_name, val_loss in val_metrics.items():
                        self.writer_validation.add_scalar(metric_name, val_loss, num_iterations)

                #if num_iterations > self.start_decay:
                #    self.adjust_lr_rate(self.generator_optimizer, name="generator")
                #    self.adjust_lr_rate(
                #        self.discriminator_optimizer, name="discriminator"
                #    )

                if num_iterations % 2000 == 0: #epoch % self.configs["SOLVER"]["CHECKPOINT_INTERVAL"] == 0:
                    end_time = time.time()
                    store_to_file = "Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}\n\n".format(
                        epoch, g_loss.item(), d_loss.item(), end_time - start_time_epoch
                    )
                    self.store_to_file(store_to_file)
                    print(
                        "Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}".format(
                            epoch, g_loss.item(), d_loss.item(), end_time - start_time_epoch
                        )
                    )

                    # Save the Entire model
                    store_to_file = "Saving model Checkpoint  ......"
                    self.store_to_file(store_to_file)
                    file_path = os.path.join(
                        self.modelCheckpoint, "{}_{}_CycleGAN_CheckPoint".format(epoch, num_iterations)
                    )
                    self.saveModelCheckPoint(epoch, file_path)
                    print("Model Saved!")

                if num_iterations % 2000 == 0: #epoch % self.configs["SOLVER"]["TESTING_INTERVAL"] == 0:
                    print("Generating testing...")
                    testing_start_time = time.time()

                    self.testing(self.generator, "neutral", "angry", epoch, num_iterations)  # testing A
                    #self.testing(self.generator, "angry", "neutral", epoch, num_iterations)  # testing B

                    testing_end_time = time.time()
                    store_to_file = "Time taken for testing Set: {}".format(
                        testing_end_time - testing_start_time
                    )
                    self.store_to_file(store_to_file)
                    print(
                        "Time taken for testing Set: {}".format(
                            testing_end_time - testing_start_time
                        )
                    )
            print(description)

        print("Training finished")

        self.writer_validation.close()
        self.writer_train.close()

    def testing(self, model, source_emo, target_emo, epoch, num_iterations):
        model.eval()

        if target_emo not in self.sp_datasets.keys():
            raise ValueError("val_class not in dataset dictionary")

        testing_dir = os.path.join(self.test_dir, source_emo)

        output_dir = os.path.join(
            self.base_conversion_dir, "{}_to_{}".format(source_emo, target_emo)
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        number_of_classes = len(self.sp_datasets.keys())
        class_enc = torch.tensor([self.emotions.index(target_emo)]).to(self.device)
        class_enc = F.one_hot(class_enc, number_of_classes).float()

        for file in os.listdir(testing_dir):
            filePath = os.path.join(testing_dir, file)
            wav, _ = librosa.load(filePath, sr=self.sampling_rate, mono=True)
            wav = preprocess.wav_padding(
                wav=wav,
                sr=self.sampling_rate,
                frame_period=self.frame_period,
                multiple=4,
            )
            f0, timeaxis, sp, ap = preprocess.world_decompose(
                wav=wav, fs=self.sampling_rate, frame_period=self.frame_period
            )

            coded_sp = preprocess.world_encode_spectral_envelop(
                sp=sp, fs=self.sampling_rate, dim=self.num_mcep
            )
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (
                coded_sp_transposed - self.sps_stats["mean"]
            ) / self.sps_stats["std"]

            log_norm_f0 = (
                np.log(f0 + 1e-10) - self.logf0_stats_emo[source_emo]["mean"]
            ) / self.logf0_stats_emo[source_emo]["std"]

            input = np.concatenate(
                (coded_sp_norm, np.expand_dims(log_norm_f0, axis=0)), axis=0
            )

            input_converted = []
            for i in range(0, input.shape[1], self.n_frames):
                segment = input[:, i : i + self.n_frames]
                if segment.shape[1] < self.n_frames:
                    # Pad the last segment if needed
                    padding = np.zeros(
                        (segment.shape[0], self.n_frames - segment.shape[1])
                    )
                    segment = np.concatenate((segment, padding), axis=1)
                segment = torch.from_numpy(segment[np.newaxis, :, :]).float()
                if torch.cuda.is_available():
                    segment = segment.cuda()
                segment_converted = model(segment, class_enc)
                input_converted.append(segment_converted.cpu().detach().numpy())

            input_converted = np.concatenate(input_converted, axis=2)
            target_length = f0.shape[0]  # Length should match the F0 contour length
            current_length = input_converted.shape[2]

            if current_length > target_length:
                input_converted = input_converted[:, :, :target_length]

            input_converted = np.squeeze(input_converted)
            coded_sp_converted_norm = input_converted[: self.num_mcep]
            f0_converted = input_converted[self.num_mcep :].squeeze(0)  
            coded_sp_converted = (
                coded_sp_converted_norm * self.sps_stats["std"] + self.sps_stats["mean"]
            )

            f0_converted_scaled = (
                f0_converted * self.logf0_stats["std"]
            ) + self.logf0_stats["mean"]
            f0_converted = np.exp(f0_converted_scaled)

            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(
                coded_sp_converted, dtype=np.double
            )
            decoded_sp_converted = preprocess.world_decode_spectral_envelop(
                coded_sp=coded_sp_converted, fs=self.sampling_rate
            )
            
            if not self.use_f0_model:
                f0_converted_n = log_norm_f0 * self.logf0_stats_emo[target_emo]["std"] + self.logf0_stats_emo[target_emo]["mean"]
                f0_converted_n = np.exp(f0_converted_n)
            

            f0_min, f0_max = 60, 500 
            f0_converted_clamped = np.clip(f0_converted, f0_min, f0_max)
            f0_smoothed = smooth_f0(f0_converted_clamped, window_size=5)
            
            wav_transformed = preprocess.world_speech_synthesis(
                f0=f0_converted,
                decoded_sp=decoded_sp_converted,
                ap=ap,
                fs=self.sampling_rate,
                frame_period=self.frame_period,
            )
            

            sf.write(
                file=os.path.join(output_dir, str(epoch) + "_" + str(num_iterations) + os.path.basename(file)),
                data=wav_transformed,
                samplerate=self.sampling_rate,
            )
            
            #visualize f0
            plt.figure()
            plt.plot(f0, label="original")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.plot(f0_converted, label="model convertion")
            plt.plot(f0_converted_n, label="normalization convertion")
            plt.legend()
            
            plt.savefig(os.path.join(output_dir, str(epoch) + "_" + str(num_iterations) + os.path.basename(file) + "_f0.png"))
            plt.close()
            
            plt.figure(figsize=(10, 4))
            D = librosa.amplitude_to_db(np.abs(sp.T), ref=np.max)
            librosa.display.specshow(D, sr=self.sampling_rate, hop_length=256, x_axis="time", y_axis="linear", cmap='coolwarm')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            plt.savefig(os.path.join(output_dir, str(epoch) + "_" + str(num_iterations) + os.path.basename(file) + "_spectrogram.png"))
            plt.close()
            
            plt.figure(figsize=(10, 4))
            D = librosa.amplitude_to_db(np.abs(decoded_sp_converted.T), ref=np.max)
            librosa.display.specshow(D, sr=self.sampling_rate, hop_length=256, x_axis="time", y_axis="linear", cmap='coolwarm')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogramconverted')
            plt.savefig(os.path.join(output_dir, str(epoch) + "_" + str(num_iterations) + os.path.basename(file) + "_spectrogram_conv.png"))
            plt.close()

    def train_classifier(self, train_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
        classifier_checkpoint = self.configs["SOLVER"]["CHECKPOINT_DIR_CLASSIFIER"]
        os.makedirs(classifier_checkpoint, exist_ok=True)

        n_epochs = 5

        for epoch in range(n_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.classifier(inputs)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                    running_loss = 0.0

            save_path = os.path.join(
                classifier_checkpoint, f"classifier_epoch_{epoch+1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.classifier.state_dict(),
                },
                save_path,
            )

        print("Finished Training")

    def load_classifier(self):
        classifier_checkpoint = self.configs["SOLVER"]["CHECKPOINT_DIR_CLASSIFIER"]
        os.path.join(classifier_checkpoint, f"classifier_epoch_{5}.pth")
        classifier_checkpoint = "./model_checkpoint/classifier/classifier_epoch_5.pth"

        checkPoint = torch.load(classifier_checkpoint, map_location=self.device)
        self.classifier.load_state_dict(state_dict=checkPoint["model_state_dict"])

    def calculate_val_error(self, val_loader):
        self.generator.eval()  # Set the generators to evaluation mode
        self.discriminator.eval()  # Set the generators to evaluation mode

        with torch.no_grad():  # Disable gradient calculation
            for i, (real, class_enc) in enumerate(val_loader):
                real = real.to(self.device).float()
                class_enc = class_enc.to(self.device)

                random_class_enc = torch.randint(
                    0, self.num_emotions, (class_enc.size(0),), device=self.device
                )
                random_class_enc = F.one_hot(
                    random_class_enc, self.num_emotions
                ).float()

                fake = self.generator(real, random_class_enc)

                cycle = self.generator(fake, class_enc)

                identity = self.generator(real, class_enc)

                d_fake_real_fake, fake_features = self.discriminator(fake)
                _, real_features = self.discriminator(real)

                feature_matching_loss = F.l1_loss(fake_features, real_features.detach())

                cycleLoss = torch.mean(torch.abs(real - cycle))

                identiyLoss = torch.mean(torch.abs(real - identity))

                generator_loss = torch.mean((1 - d_fake_real_fake) ** 2)

                tv_loss = (
                    self.total_variation_loss_spectrogram(fake)
                    + self.total_variation_loss_spectrogram(cycle)
                    + self.total_variation_loss_spectrogram(identity)
                ) / 3

                # Emotion classification loss for the generator
                predicted_emotion = self.classifier(fake)
                generator_emotion_loss = F.cross_entropy(
                    predicted_emotion, random_class_enc.argmax(dim=1)
                )

                g_loss = (
                    generator_loss
                    + self.cycle_loss_lambda * cycleLoss
                    + self.identity_loss_lambda * identiyLoss
                    + self.emotion_loss_lambda * generator_emotion_loss
                    + self.feature_loss_lambda * feature_matching_loss
                    + tv_loss * self.tv_loss_lambda
                )

                d_real_real_fake, _ = self.discriminator(real)
                d_fake_real_fake, _ = self.discriminator(
                    fake.detach()
                )  # detach to avoid backprop through generator

                d_loss_real = torch.mean((1 - d_real_real_fake) ** 2)
                d_loss_fake = torch.mean((0 - d_fake_real_fake) ** 2)
                d_loss = (d_loss_real + d_loss_fake) / 2.0

                self.loss_tracker.update(
                    [
                        g_loss.item(),
                        d_loss.item(),
                        cycleLoss.item(),
                        identiyLoss.item(),
                        generator_emotion_loss.item(),
                        feature_matching_loss.item(),
                        tv_loss.item(),
                    ]
                )

        self.generator.train()
        self.discriminator.train()
        last_n = 300
        return (
            self.loss_tracker.get_last_average("g_loss", last_n),
            self.loss_tracker.get_last_average("d_loss", last_n),
            self.loss_tracker.get_last_average("g_cycle_loss", last_n),
            self.loss_tracker.get_last_average("g_id_loss", last_n),
            self.loss_tracker.get_last_average("g_emotion_loss", last_n),
            self.loss_tracker.get_last_average("g_feature_matching_loss", last_n),
            self.loss_tracker.get_last_average("g_tv_loss", last_n),
        )

    def total_variation_loss_spectrogram(self, spectrogram):
        time_variation = (
            torch.abs(spectrogram[:, :, 1:] - spectrogram[:, :, :-1]).sum()
            / spectrogram.shape[2]
        )
        return time_variation

    def savePickle(self, variable, fileName):
        with open(fileName, "wb") as f:
            pickle.dump(variable, f)

    def loadPickleFile(self, fileName):
        with open(fileName, "rb") as f:
            return pickle.load(f)

    def load_config(self, config_file="config.yaml"):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        return config

    def store_to_file(self, doc):
        doc = doc + "\n"
        with open(self.file_name, "a") as myfile:
            myfile.write(doc)

    def saveModelCheckPoint(self, epoch, PATH):
        torch.save(
            {
                "epoch": epoch,
                "model_gen_state_dict": self.generator.state_dict(),
                "model_discriminator": self.discriminator.state_dict(),
                "generator_optimizer": self.generator_optimizer.state_dict(),
                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            },
            PATH,
        )

    def loadModel(self, PATH):
        checkPoint = torch.load(PATH)
        self.generator.load_state_dict(state_dict=checkPoint["model_gen_state_dict"])
        self.discriminator.load_state_dict(state_dict=checkPoint["model_discriminator"])
        self.generator_optimizer.load_state_dict(
            state_dict=checkPoint["generator_optimizer"]
        )
        self.discriminator_optimizer.load_state_dict(
            state_dict=checkPoint["discriminator_optimizer"]
        )
        epoch = int(checkPoint["epoch"])
        return epoch + 1

    def load_emotion_data(self, cache_folder, emotions):
        coded_sps = {}
        f0s = {}

        for emotion in emotions:
            coded_sp_file = os.path.join(
                cache_folder, f"coded_sps_{emotion}_norm.pickle"
            )
            f0_file = os.path.join(cache_folder, f"f0_norm_{emotion}.pickle")

            try:
                with open(coded_sp_file, "rb") as f:
                    coded_sps[emotion] = pickle.load(f)
                with open(f0_file, "rb") as f:
                    f0s[emotion] = pickle.load(f)
            except FileNotFoundError as e:
                print(f"File not found: {e}")
        return coded_sps, f0s

    def load_statistics(self, cache_folder):
        """Load the logf0 and mcep normalization statistics from the saved files and return them as dictionaries."""

        logf0_norm_file = os.path.join(cache_folder, "logf0s_normalization.npz")
        mcep_norm_file = os.path.join(cache_folder, "mcep_normalization.npz")

        logf0_stats = {}
        mcep_stats = {}

        if os.path.exists(logf0_norm_file):
            logf0_norm_data = np.load(logf0_norm_file)
            logf0_stats["mean"] = logf0_norm_data["mean"]
            logf0_stats["std"] = logf0_norm_data["std"]
        else:
            raise FileNotFoundError(
                f"logf0 normalization file not found at {logf0_norm_file}"
            )

        if os.path.exists(mcep_norm_file):
            mcep_norm_data = np.load(mcep_norm_file)
            mcep_stats["mean"] = mcep_norm_data["mean"]
            mcep_stats["std"] = mcep_norm_data["std"]
        else:
            raise FileNotFoundError(
                f"mcep normalization file not found at {mcep_norm_file}"
            )
            
            
            
        logf0_stats_emo = {emotion: {} for emotion in self.emotions}


        for emotion in self.emotions:
            logf0_norm_file_emo = os.path.join(cache_folder, f"logf0s_{emotion}_normalization.npz")
            if os.path.exists(logf0_norm_file_emo):
                logf0_norm_data_emo = np.load(logf0_norm_file_emo)
                logf0_stats_emo[emotion]["mean"] = logf0_norm_data_emo["mean"]
                logf0_stats_emo[emotion]["std"] = logf0_norm_data_emo["std"]
            else:
                raise FileNotFoundError(
                    f"logf0 normalization file for {emotion} not found at {logf0_norm_file_emo}")

        return logf0_stats, mcep_stats, logf0_stats_emo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CycleGAN using source dataset and target dataset"
    )

    config_file = "./config.yaml"
    resume_training_at = None


    parser.add_argument(
        "--config_file",
        type=str,
        help="location of the config file",
        default=config_file,
    )
    parser.add_argument(
        "--resume_training",
        type=str,
        help="Location of the pre-trained model to resume training",
        default=resume_training_at,
    )

    argv = parser.parse_args()
    config_file = argv.config_file
    resume_training_at = argv.resume_training

    cycleGAN = CycleGANTraining(
        config_file=config_file, restart_training_at=resume_training_at
    )
    cycleGAN.train()
