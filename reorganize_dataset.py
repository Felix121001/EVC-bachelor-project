import soundfile as sf
import numpy as np
import os
import shutil
import argparse


def convert_to_int16(directory):
    for folder in os.listdir(directory):
        directory_path = os.path.join(directory, folder)
        print(folder)
        if not os.path.isdir(directory_path):
            print("Not a directory")
            continue

        new_folder = os.path.join(directory, folder + "-int16")
        # Create the new directory if it doesn't exist
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        for file in os.listdir(directory_path):
            if file.endswith(".wav"):
                file_path = os.path.join(directory_path, file)
                data, samplerate = sf.read(file_path, dtype="float32")
                int_data = (data * np.iinfo(np.int16).max).astype(np.int16)
                output_file_path = os.path.join(new_folder, "converted_" + file)
                sf.write(output_file_path, int_data, samplerate)


def organize_ravdess_by_emotion(
    dataset_path, target_path, remove_original_folders=False
):
    emotions = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    for emotion in emotions.values():
        emotion_dir = os.path.join(target_path, emotion)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)

    for actor_folder in os.listdir(dataset_path):
        actor_folder_path = os.path.join(dataset_path, actor_folder)
        if os.path.isdir(actor_folder_path):
            for filename in os.listdir(actor_folder_path):
                if filename.endswith(".wav"):
                    emotion_code = filename.split("-")[2]
                    emotion = emotions.get(emotion_code, "unknown")

                    src_file_path = os.path.join(actor_folder_path, filename)
                    dest_file_path = os.path.join(target_path, emotion, filename)
                    shutil.move(src_file_path, dest_file_path)

            if remove_original_folders:
                os.rmdir(actor_folder_path)


def organize_CREMA_D_by_emotion(source_folder, target_folder):
    emotions = {
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fearful",
        "HAP": "happy",
        "NEU": "neutral",
        "SAD": "sad",
    }

    for emotion_name in emotions.values():
        emotion_dir = os.path.join(target_folder, emotion_name)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)

    for filename in os.listdir(source_folder):
        if filename.endswith(".wav"):
            parts = filename.split("_")
            if len(parts) > 2 and parts[2] in emotions:
                emotion_code = parts[2]
                emotion = emotions[emotion_code]

                # Move the file to the corresponding emotion directory
                src_file_path = os.path.join(source_folder, filename)
                dest_file_path = os.path.join(target_folder, emotion, filename)
                shutil.move(src_file_path, dest_file_path)


def organize_emodb_by_emotion(source_folder, target_folder):
    emotions = {
        "W": "angry",
        "L": "bored",
        "E": "disgust",
        "A": "fearful",
        "F": "happy",
        "T": "sad",
        "N": "neutral",
    }

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for emotion_name in emotions.values():
        emotion_dir = os.path.join(target_folder, emotion_name)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)

    for filename in os.listdir(source_folder):
        if filename.endswith(".wav") and len(filename) > 5:
            emotion_code = filename[5]
            emotion = emotions[emotion_code]

            src_file_path = os.path.join(source_folder, filename)
            dest_file_path = os.path.join(target_folder, emotion, filename)
            shutil.move(src_file_path, dest_file_path)


def reorganize_emov_db_by_emotion(source_folder, target_folder):
    emotions = {
        "Angry": "angry",
        "Disgusted": "disgust",
        "Amused": "happy",
        "Sleepy": "sleepy",
        "Neutral": "neutral",
    }

    for actor_folder in os.listdir(source_folder):
        actor_folder_path = os.path.join(source_folder, actor_folder)

        if os.path.isdir(actor_folder_path) and actor_folder.endswith("-int16"):
            original_emotion_name = actor_folder.split("_")[1][:-6]

            new_emotion_name = emotions[original_emotion_name]

            new_emotion_dir = os.path.join(target_folder, new_emotion_name)
            if not os.path.exists(new_emotion_dir):
                os.makedirs(new_emotion_dir)

            # Move all .wav files from the current actor's folder to the new emotion folder
            for filename in os.listdir(actor_folder_path):
                if filename.endswith(".wav"):
                    src_file_path = os.path.join(actor_folder_path, filename)
                    dest_file_path = os.path.join(new_emotion_dir, filename)
                    shutil.move(src_file_path, dest_file_path)


def organize_iemocap_by_emotion(iemocap_dir, target_dir):
    # Emotion mapping
    emotion_dict = {"ang": "angry", "sad": "sad", "hap": "happy", "neu": "neutral"}

    for session in os.listdir(iemocap_dir):
        if not session.startswith("Session"):
            continue

        session_dir = os.path.join(iemocap_dir, session)
        annotations_dir = os.path.join(session_dir, "dialog", "EmoEvaluation")
        wav_dir = os.path.join(session_dir, "sentences", "wav")

        for foldername in os.listdir(wav_dir):
            if not foldername.startswith("Ses"):
                continue

            subsession_dir = os.path.join(wav_dir, foldername)
            for filename in os.listdir(subsession_dir):
                if not filename.endswith(".wav"):
                    continue

                emotion_label = get_emotion_from_wav(filename, annotations_dir)
                emotion = emotion_dict.get(emotion_label, "unknown")

                emotion_dir = os.path.join(target_dir, emotion)
                if not os.path.exists(emotion_dir):
                    os.makedirs(emotion_dir)

                src_file = os.path.join(subsession_dir, filename)
                dest_file = os.path.join(emotion_dir, filename)
                shutil.copy2(src_file, dest_file)

        print(f"{session} completed.")


def get_emotion_from_wav(filename, annotations_dir):
    label_file_name = filename[:-9] + ".txt"
    label_file_path = os.path.join(annotations_dir, label_file_name)

    with open(label_file_path, "r") as label_file:
        for row in label_file:
            if row.startswith("[") and filename[:-4] in row:
                split = row.split("\t")
                category = split[2]
                return (
                    category if category in ["ang", "sad", "hap", "neu"] else "unknown"
                )

    return "unknown"


def combine_emotions_from_datasets(data_dir, combined_dir, datasets, emotions):
    available_emotions = {
        "CREMA-D": ["angry", "disgust", "fearful", "happy", "neutral", "sad"],
        "EmoDB": ["angry", "bored", "disgust", "fearful", "happy", "neutral", "sad"],
        "EmoV-DB": ["angry", "disgust", "happy", "neutral", "sleepy"],
        "IEMOCAP_test": ["angry", "happy", "neutral", "sad"],
        "RAVDESS": [
            "angry",
            "calm",
            "disgust",
            "fearful",
            "happy",
            "neutral",
            "sad",
            "surprised",
        ],
    }

    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)

    for emotion in emotions:
        emotion_dir = os.path.join(combined_dir, emotion)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)

    for dataset in datasets:
        if dataset in available_emotions:
            dataset_dir = os.path.join(data_dir, dataset)
            for emotion in emotions:
                if emotion in available_emotions[dataset]:
                    emotion_dir = os.path.join(dataset_dir, emotion)
                    if os.path.exists(emotion_dir):
                        for filename in os.listdir(emotion_dir):
                            if filename.endswith(".wav"):
                                src_file = os.path.join(emotion_dir, filename)
                                dest_file = os.path.join(
                                    combined_dir, emotion, f"{dataset}_{filename}"
                                )
                                shutil.copy2(src_file, dest_file)


if __name__ == "__main__":
    print("Converting...")

    datasets = ["CREMA-D", "EmoDB", "EmoV-DB", "IEMOCAP_test", "RAVDESS"]
    emotions = ["angry", "sad", "neutral", "happy"]


    parser = argparse.ArgumentParser(
        description="Preprocesses the training data for the Voice Conversion Model"
    )

    config_file = "./config.yaml"

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset to preprocess",
        default=None,
    )
    
    parser.add_argument(
        "--source_path",
        type=str,
        help="Path to the dataset",
        default=None,
    )
    
    parser.add_argument(
        "--target_path",
        type=str,
        help="Path to the dataset",
        default=None,
    )

    argv = parser.parse_args()
    dataset_name = argv.dataset_name
    source_path = argv.source_path
    target_path = argv.target_path
    
    
    if not os.path.exists('./data/'):
        os.makedirs('./data/')
    
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    
    match dataset_name:
        case "CREMA-D":
            organize_CREMA_D_by_emotion(source_path, target_path)
        case "EmoDB":
            organize_emodb_by_emotion(source_path, target_path)
        case "EmoV-DB":
            reorganize_emov_db_by_emotion(source_path, target_path)
        case "IEMOCAP":
            organize_iemocap_by_emotion(source_path, target_path)
        case "RAVDESS":
            organize_ravdess_by_emotion(source_path, target_path)
        case "combine":
            combine_emotions_from_datasets(source_path, target_path, datasets, emotions)
            

    