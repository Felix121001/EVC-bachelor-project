## About This Project
This repository hosts the code for the bachelor project titled "Exploring Cycle GAN & Diffusion Models in the Context of Emotional Voice Conversion."

The primary dataset used for this project is IEMOCAP, which requires signing in to access at: IEMOCAP. Additionally, the project is compatible with other datasets, such as:

CREMA-D: CREMA-D Repository
EmoDB: EmoDB Download
EmoV-DB: EmoV-DB
RAVDESS: RAVDESS on Kaggle

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
python reorganize_dataset.py --dataset_name IEMOCAP --source_path "DIR" --target_path "./data/IEMOCAP"
```
For CREMA-D:
```
python reorganize_dataset.py --dataset_name CREMA-D --source_path "DIR" --target_path "./data/CREMA-D"
```
For EmoDB:
```
python reorganize_dataset.py --dataset_name EmoDB --source_path "DIR" --target_path "./data/EmoDB"
```
For EmoV-DB:
```
python reorganize_dataset.py --dataset_name EmoV-DB --source_path "DIR" --target_path "./data/EmoV-DB"
```
For RAVDESS:
```
python reorganize_dataset.py --dataset_name RAVDESS --source_path "DIR" --target_path "./data/RAVDESS"
```
If training is to be done with a combined dataset run:
```
python reorganize_dataset.py --dataset_name combine --source_path "./data/" --target_path "./data/combined_dataset"
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











