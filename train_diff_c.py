import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import torch
import torch.optim as optim
from tqdm import tqdm
from Diffusion.diffusion_module import Unet_conditional
import sys
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
from trainingDataset import AudioDataset, ProcessedAudioDatasetCombined
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import preprocess
import soundfile as sf
import argparse
import yaml
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import librosa
from GAN.utils import safe_coded_spectrogram, save_2d_vector_as_image

os.environ['KMP_DUPLICATE_LIB_OK']='True'


# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DiffusionModel(nn.Module):
    def __init__(
        self,
        timesteps,
        num_classes,
        class_emb_dim,
        channels,
        height,
        width,
        ckpt_path,
        device,
        beta_min,
        beta_max,
    ):
        super(DiffusionModel, self).__init__()
        self.device = device
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_min, beta_max, timesteps, device=device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cat(
            (torch.tensor([1.0], device=device), torch.cumprod(self.alpha, 0)[:-1]),
            dim=0,
        )

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.one_minus_sqrt_alpha_bar = torch.sqrt(1 - self.alpha_bar)

        self.unet = Unet_conditional(
            num_classes=num_classes,
            class_emb_dim=class_emb_dim,
            channels=channels,
            height=height,
            width=width,
            device=device,
        ).to(device)

        self.ckpt_path = ckpt_path
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(self.ckpt_path, "ckpt_10.pth")
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        if glob.glob(f"{model_path}"):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.unet.load_state_dict(checkpoint["model_state_dict"])
            print("Model loaded successfully.")
        else:
            print("No checkpoint file found to load.")

    def generate_noise(self, key, x_0, t):
        torch.manual_seed(key)
        noise = torch.randn_like(x_0, device=self.device)
        reshaped_sqrt_alpha_bar_t = (
            self.sqrt_alpha_bar[t].view(-1, 1, 1, 1) 
        )
        reshaped_one_minus_sqrt_alpha_bar_t = (
            self.one_minus_sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        )
        noisy_image = (
            reshaped_sqrt_alpha_bar_t * x_0
            + reshaped_one_minus_sqrt_alpha_bar_t * noise
        )
        return noisy_image, noise

    def generate_timestamp(self, key, num):
        torch.manual_seed(key)
        return torch.randint(
            low=0,
            high=self.timesteps,
            size=(num,),
            dtype=torch.int32,
            device=self.device,
        )

    def ddpm(self, x_t, pred_noise, t):
        alpha_t = self.alpha[t]
        alpha_t_bar = self.alpha_bar[t]
        eps_coef = (1 - alpha_t) / torch.sqrt(1 - alpha_t_bar)
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - eps_coef * pred_noise)
        var = self.beta[t]
        z = torch.normal(mean=0.0, std=1.0, size=x_t.shape, device=self.device)
        return mean + torch.sqrt(var) * z


class DiffusionPipeline:
    def __init__(self, config_file, resume_training_at):
        self.configs = self.load_config(config_file)

        self.epochs = self.configs["SOLVER"]["MAX_EPOCHS"]
        self.batch_size = self.configs["SOLVER"]["BATCH_SIZE"]
        self.timesteps = self.configs["SOLVER"]["TIMESTEPS"]
        self.class_emb_dim = self.configs["SOLVER"]["CLASS_EMB_DIM"]
        self.current_run_name = self.configs["SOLVER"]["RUN_NAME"]
        self.channels = self.configs["SOLVER"]["CHANNELS"]
        self.ckpt_path = self.configs["SOLVER"]["CHECKPOINT_DIR"]
        self.ckpt_path = os.path.join(self.ckpt_path, self.current_run_name)
        print("Current run name:", self.current_run_name)
        self.lr = self.configs["SOLVER"]["LR"]
        self.beta_min = self.configs["SOLVER"]["BETA_MIN"]
        self.beta_max = self.configs["SOLVER"]["BETA_MAX"]
        self.beta_min = float(self.beta_min)
        self.beta_max = float(self.beta_max)

        if resume_training_at is not None:
            self.ckpt_path = resume_training_at

        self.cache_dir = self.configs["DATASET"]["CACHE"]
        self.emotions = self.configs["DATASET"]["EMOTIONS"]
        self.test_dir = os.path.join(
            self.configs["DATASET"]["TRAINING_DATASET"], "test"
        )
        self.num_classes = len(self.emotions)

        self.n_frames = self.configs["INPUT"]["N_FRAMES"]
        self.num_mcep = self.configs["INPUT"]["NUM_MCEP"]
        self.sampling_rate = self.configs["INPUT"]["SAMPLING_RATE"]
        self.frame_period = self.configs["INPUT"]["FRAME_PERIOD"]

        os.makedirs("converted_sound", exist_ok=True)
        self.base_conversion_dir = os.path.join(
            "converted_sound", self.current_run_name
        )
        os.makedirs(self.base_conversion_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        self.model = DiffusionModel(
            self.timesteps,
            self.num_classes,
            self.class_emb_dim,
            self.channels,
            self.num_mcep + 1,
            self.n_frames,
            self.ckpt_path,
            self.device,
            self.beta_min,
            self.beta_max,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.start_epoch = 0
        self.train_loader, self.test_loader = self.get_train_loader(
            self.batch_size, self.n_frames
        )

        run_directory = os.path.join("runs", self.current_run_name)
        validation_directory = os.path.join(run_directory, "validation")
        train_directory = os.path.join(run_directory, "train")
        os.makedirs(run_directory, exist_ok=True)
        os.makedirs(validation_directory, exist_ok=True)
        os.makedirs(train_directory, exist_ok=True)
        self.writer_train = SummaryWriter(train_directory)
        self.writer_test = SummaryWriter(validation_directory)

    def train(self):

        #self.testing("angry", "neutral", 0)
        
        comb_losses = []
        for epoch in range(self.start_epoch, self.epochs):
            pbar = tqdm(
                total=len(self.train_loader),
                desc="Epoch {}".format(epoch),
                mininterval=1,
            )
            losses = []
            for i, (batch, _class) in enumerate(self.train_loader):
                
                batch = batch.to(self.device)
                _class = _class.to(self.device)
                if len(batch.shape) == 3:
                    batch = batch.unsqueeze(1)

                loss, sp = self.train_step(batch, _class)
                losses.append(loss)

                if i % 10 == 0:
                    description = f"Epoch {epoch+1}/{self.epochs} | Loss: {loss}"
                    pbar.set_description(description)

                if (i + 1) % 200 == 0:
                    self.writer_train.add_scalar(
                        "loss",
                        torch.mean(
                            torch.tensor(losses[:-50], device=self.device)
                        ).item(),
                        epoch * len(self.train_loader) + i,
                    )
                    self.validate(
                        epoch, indx=epoch * len(self.train_loader) + i, log=False
                    )
                    
                output_dir = os.path.join(self.base_conversion_dir, "testing")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                plt.figure(figsize=(10, 4))
                sp = sp.squeeze().cpu().detach().numpy()
                D = librosa.amplitude_to_db(np.abs(sp.T), ref=np.max)
                librosa.display.specshow(D, sr=self.sampling_rate, hop_length=256, x_axis="time", y_axis="linear", cmap='coolwarm')
                plt.colorbar(format='%+2.0f dB')
                plt.title('Spectrogram')
                plt.savefig(os.path.join(output_dir, str(epoch) + "_" + str(i) + "_spectrogram.png"))
                plt.close()
                
                

            pbar.close()
            avg_loss = torch.mean(torch.tensor(losses, device=self.device)).item()
            print(
                f"Average training loss for epoch {epoch+1}/{self.epochs}: {avg_loss}"
            )
            comb_losses.extend(losses)

            if epoch % 1 == 0:
                self.validate(
                    epoch, (epoch + 1) * len(self.train_loader), add_scalar=False
                )
                self.testing("angry", "neutral", epoch)

            if epoch % 2 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.unet.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": avg_loss,
                    },
                    os.path.join(self.ckpt_path, f"ckpt_test_{epoch+1}.pth"),
                )
                print("Model saved successfully.")
        # self.plot_losses(comb_losses)

    def train_step(self, batch, _class):
        rng = torch.randint(0, 100000, (1,), device=self.device).item()
        tsrng = torch.randint(0, 100000, (1,), device=self.device).item()

        timestep_values = self.model.generate_timestamp(tsrng, batch.shape[0])
        noised_image, noise = self.model.generate_noise(
            rng, batch, timestep_values.int()
        )
        self.optimizer.zero_grad()
        prediction = self.model.unet(noised_image, timestep_values.float(), _class)
        denoised_spectrogram = noised_image - prediction
        loss_value = torch.mean((noise - prediction) ** 2)
        loss_value.backward()
        self.optimizer.step()
        return loss_value.item(), denoised_spectrogram.detach()

    def validate(self, epoch, indx, log=True, add_scalar=True):
        with torch.no_grad():
            losses = []
            for i, (batch, _class) in enumerate(self.test_loader):
                batch = batch.to(self.device)
                _class = _class.to(self.device)
                if len(batch.shape) == 3:
                    batch = batch.unsqueeze(1)
                rng = torch.randint(0, 100000, (1,), device=self.device).item()
                tsrng = torch.randint(0, 100000, (1,), device=self.device).item()

                timestep_values = self.model.generate_timestamp(tsrng, batch.shape[0])
                noised_image, noise = self.model.generate_noise(
                    rng, batch, timestep_values.int()
                )
                prediction = self.model.unet(
                    noised_image, timestep_values.float(), _class
                )
                loss_value = torch.mean((noise - prediction) ** 2)
                losses.append(loss_value.item())

                # if i == 0:
                #    self.save_gif(self, [noised_image[0].squeeze().cpu(), prediction[0].squeeze().cpu(), batch[0][0].squeeze().cpu()], f"./Diffusion/results/epoch_{epoch}.gif")

            avg_loss = torch.mean(torch.tensor(losses, device=self.device)).item()
            if add_scalar:
                self.writer_test.add_scalar("loss_val", avg_loss, indx)
            if log:
                print(
                    f"Average test loss for epoch {epoch+1}/{self.epochs}: {avg_loss}"
                )

    def transforme_data(self, data, target_class):
        print("Transforming data")
        data = data.to(self.device)
        target_class = target_class.to(self.device)

        rng = torch.randint(0, 100000, (1,), device=self.device).item()
        tsrng = torch.randint(0, 100000, (1,), device=self.device).item()
        timestep_values = self.model.generate_timestamp(tsrng, data.shape[0])
        for i in range(self.timesteps - 1):
            if i % 50 == 0:
                timestep_va = self.model.generate_timestamp(tsrng, data.shape[0])
                noised_data, _ = self.model.generate_noise(
                    rng, data, timestep_va.int()
                )
                save_2d_vector_as_image(noised_data[0][0].cpu().detach().numpy(), self.sampling_rate,  filename=f"./Diffusion/results/noise{i}.png")
                
        noised_data, _ = self.model.generate_noise(
            rng, data, timestep_values.int()
        )

        x = noised_data

        for i in tqdm(range(self.timesteps - 1)):
            t = torch.tensor([self.timesteps - i - 1], dtype=torch.int32).to(self.device)

            pred_noise = self.model.unet(x, t, target_class).to(self.device)
            x = self.model.ddpm(x, pred_noise, t)

            if i % 50 == 0:
                save_2d_vector_as_image(noised_data[0][0].cpu().detach().numpy(), self.sampling_rate,  filename=f"./Diffusion/results/denoise{i}.png")

        return x[0]
    
    
    def reverse_diffusion(self, data, class_enc):
        class_enc = class_enc.to(self.device)
        x = torch.randn(data.shape[0], 1, data.shape[1], data.shape[2]).to(self.device) 
        
        for i in tqdm(range(self.timesteps - 1)):
            t = torch.tensor([self.timesteps - i - 1], dtype=torch.int32).to(self.device)
            
            pred_noise = self.model.unet(x, t, class_enc)


            x = self.reverse_step(x, pred_noise, t)

        print("x shape", x.shape)
        return x[0]

    def reverse_step(self, x_t, pred_noise, t):
        alpha_t = self.model.alpha[t]
        alpha_t_bar = self.model.alpha_bar[t]

        # Compute the mean for the reverse step
        eps_coef = (1 - alpha_t) / torch.sqrt(1 - alpha_t_bar)
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - eps_coef * pred_noise)

        # No additional noise should be added during the reverse process
        return mean

    def testing(self, source_emo, target_emo, epoch):
        if target_emo not in self.sp_datasets.keys():
            raise ValueError("val_class not in dataset dictionary")
        target_emo = source_emo  # for now to test on same emotion

        testing_dir = os.path.join(self.test_dir, source_emo)

        output_dir = os.path.join(
            self.base_conversion_dir, "{}_to_{}".format(source_emo, target_emo)
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        class_enc = torch.tensor(
            [self.emotions.index(target_emo)], dtype=torch.float32
        ).to(self.device)

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
                np.log(f0 + 1e-10) - self.logf0_stats["mean"]
            ) / self.logf0_stats["std"]

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
                with torch.no_grad():
                    #segment_converted = self.reverse_diffusion(segment, class_enc)
                    segment_converted = self.transforme_data(segment, class_enc)
                input_converted.append(segment_converted.cpu().detach().numpy())
                

            input_converted = np.concatenate(input_converted, axis=2)
            target_length = f0.shape[0]  # Length should match the F0 contour length
            current_length = input_converted.shape[2]

            if current_length > target_length:
                input_converted = input_converted[:, :, :target_length]

            input_converted = np.squeeze(input_converted)
            coded_sp_converted_norm = input_converted[: self.num_mcep]
            f0_converted = input_converted[self.num_mcep :].squeeze(
                0
            )  # last element is f0_converted
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

            wav_transformed = preprocess.world_speech_synthesis(
                f0=f0_converted,
                decoded_sp=decoded_sp_converted,
                ap=ap,
                fs=self.sampling_rate,
                frame_period=self.frame_period,
            )

            sf.write(
                file=os.path.join(output_dir, str(epoch) + os.path.basename(file)),
                data=wav_transformed,
                samplerate=self.sampling_rate,
            )
            
            
            #visualize f0
            plt.figure()
            plt.plot(f0, label="original")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.plot(f0_converted, label="converted")
            plt.legend()
            
            plt.savefig(os.path.join(output_dir, str(epoch) + "_" + str(0) + os.path.basename(file) + "_f0.png"))
            plt.close()
            
            plt.figure(figsize=(10, 4))
            D = librosa.amplitude_to_db(np.abs(sp.T), ref=np.max)
            librosa.display.specshow(D, sr=self.sampling_rate, hop_length=256, x_axis="time", y_axis="linear", cmap='coolwarm')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            plt.savefig(os.path.join(output_dir, str(epoch) + "_" + str(0) + os.path.basename(file) + "_spectrogram.png"))
            plt.close()
            
            plt.figure(figsize=(10, 4))
            D = librosa.amplitude_to_db(np.abs(decoded_sp_converted.T), ref=np.max)
            librosa.display.specshow(D, sr=self.sampling_rate, hop_length=256, x_axis="time", y_axis="linear", cmap='coolwarm')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogramconverted')
            plt.savefig(os.path.join(output_dir, str(epoch) + "_" + str(0) + os.path.basename(file) + "_spectrogram_conv.png"))
            plt.close()
            

    @staticmethod
    def save_gif(self, img_list, path="", interval=200):
        imgs = [(np.array(im) + 1) * 127.5 for im in img_list]
        imgs = [np.clip(im, 0, 255).astype(np.uint8) for im in imgs]
        imgs = [Image.fromarray(im) for im in imgs]
        imgs[0].save(
            path,
            save_all=True,
            append_images=imgs[1:],
            optimize=False,
            duration=interval,
            loop=0,
        )

    def load_pickle_file(self, file_name):
        with open(file_name, "rb") as f:
            return pickle.load(f)

    def get_train_loader(self, mini_batch_size=32, n_frames=128):
        self.sp_datasets, self.f0_datasets = self.load_emotion_data(
            self.cache_dir, self.emotions
        )
        self.logf0_stats, self.sps_stats = self.load_statistics(self.cache_dir)

        dataset = ProcessedAudioDatasetCombined(
            self.sp_datasets, self.f0_datasets, n_frames=n_frames
        )

        n_samples = len(dataset)
        print("Number of training samples: {}".format(n_samples - 100))
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [n_samples - 100, 100]
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=mini_batch_size,
            shuffle=True,
            drop_last=False,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=mini_batch_size,
            shuffle=True,
            drop_last=False,
        )

        return train_loader, test_loader

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

        return logf0_stats, mcep_stats

    def plot_losses(self, losses):
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Training Loss")
        plt.title("Training Loss Over Iterations")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    def load_config(self, config_file="config.yaml"):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        return config


# Main script execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CycleGAN using source dataset and target dataset"
    )

    config_file = "./config_diff.yaml"
    resume_training_at = './model_checkpoint/conditional_diffusion/Diffusion_200_timesteps_0.02_lr_1.0e-05_testing/ckpt_test_15.pth'

    parser.add_argument(
        "--config_file",
        type=str,
        help="location of the config file",
        default=config_file,
    )
    parser.add_argument(
        "--resume_training_at",
        type=str,
        help="Location of the pre-trained model to resume training",
        default=resume_training_at,
    )

    pipeline = DiffusionPipeline(config_file, resume_training_at)
    pipeline.train()
