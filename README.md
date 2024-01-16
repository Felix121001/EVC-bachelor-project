## About This Project
This repository hosts the code for the bachelor project titled "Exploring Cycle GAN & Diffusion Models in the Context of Emotional Voice Conversion."

The primary dataset used in this project is IEMOCAP, which requires signing in to access: [IEMOCAP](https://sail.usc.edu/iemocap/). Additionally, the project is compatible with other datasets, such as:
- CREMA-D: [CREMA-D Repository](https://github.com/CheyneyComputerScience/CREMA-D)
- EmoDB: [EmoDB Download](http://www.emodb.bilderbar.info/download/)
- EmoV-DB: [EmoV-DB](https://www.openslr.org/115/)
- RAVDESS: [RAVDESS on Kaggle](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)


The config.yaml (for the GAN model) or config_diff.yaml (for the Diffusion model) can be edited to change various settings, such as hyperparameters for training, dataset choices, emotion domains to be learned, etc.

## Requirements
Below are the package versions of the most crucial libraries used on python 3.11:


- torchaudio=2.1.1+cu121 
- torchvision=0.16.1+cu121 
- numpy=1.26.2 
- scipy=1.11.4 
- librosa=0.10.1
- soundfile=0.12.1 
- matplotlib=3.8.2 
- scikit-learn=1.3.2 
- tensorboard=2.15.1 
- pyworld=0.3.4


## Reorganize Datasets 
The initial preprocessing step involves categorizing all audio files from the selected dataset based on their emotion. These files are then organized into separate folders as shown below:
```
IEMOCAP
  |- angry  
        |- Ses01F_impro01  
        |- Ses01F_impro02  
        |- ...  
  |- happy
        |- Ses05F_impro01
        |- Ses05F_impro02
        |- ...
  |-...
```

To reorganize the IEMOCAP dataset, run: 
```
python reorganize_dataset.py --dataset_name IEMOCAP --source_path "DIR" --target_path ./data/IEMOCAP
```
For CREMA-D:
```
python reorganize_dataset.py --dataset_name CREMA-D --source_path "DIR" --target_path ./data/CREMA-D
```
For EmoDB:
```
python reorganize_dataset.py --dataset_name EmoDB --source_path "DIR" --target_path ./data/EmoDB
```
For EmoV-DB:
```
python reorganize_dataset.py --dataset_name EmoV-DB --source_path "DIR" --target_path ./data/EmoV-DB
```
For RAVDESS:
```
python reorganize_dataset.py --dataset_name RAVDESS --source_path "DIR" --target_path ./data/RAVDESS
```
If training is to be done with a combined dataset run:
```
python reorganize_dataset.py --dataset_name combine --source_path "./data/" --target_path ./data/combined_dataset
```
## Preprocessing 


The subsequent preprocessing step involves extracting F0, spectrogram, and aperiodicities, which are saved in the ./cache folder. This step is performed once to avoid repetition at the start of each training session. The default configuration file is ./config.yaml:
```
python preprocess_training.py --config_file ./config.yaml
```

## Train
To load a checkpoint the argument --resume_training can be used. To train the GAN model, use:
```
python train.py --config_file ./config.yaml
```
To train Diffusion model, use:
```
pyhton train_diff.py --config_file ./config_diff.yaml
```
To resume training at checkpoint add argument:
```
--resume_training <DIR>
```
## Results from training
If TensorBoard is installed the loss during training can be visualized by the command:
```
tensorboard --logdir runs
```









