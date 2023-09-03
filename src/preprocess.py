# import system libs
import os
import pathlib
from time import perf_counter
from typing import List

# import data handling tools
import pandas as pd

# import Deep learning Libraries
import tensorflow as tf

from config import Config


def generate_data_paths_labels(data_dir):
    """
    Generates file paths and labels for the data in the specified directory.

    Args:
        data_dir: The directory path where the data is located.

    Returns:
        filepaths (List[str]): A list of file paths for the data files.
        labels (List[str]): A list of corresponding labels for the data files.

    """
    filepaths = []
    labels = []

    # Get the list of subdirectories in the data directory
    folds = os.listdir(data_dir)[:2]

    # Iterate over the subdirectories
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)

        # Iterate over the files in each subdirectory
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)

    return filepaths, labels


def create_image_dataframe(filepaths: List[str]):
    """
    Creates a DataFrame with filepaths and corresponding labels.

    Args:
        filepaths (List[str]): A list of filepaths.

    Returns:
        pd.DataFrame: A DataFrame containing the filepaths and labels
    """

    labels = [pathlib.Path(filepath).parent.name for filepath in filepaths]

    filepath_series = pd.Series(filepaths, name="Filepath").astype(str)
    labels_series = pd.Series(labels, name="Label")

    # Concatenate filepaths and labels
    df = pd.concat([filepath_series, labels_series], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1, random_state=Config.seed).reset_index(drop=True)

    return df


def create_gen(train_df, val_df, test_df):
    """
    Create image data generators for training, validation, and testing.

    Returns:
        train_generator (ImageDataGenerator): Image data generator for training data.
        test_generator (ImageDataGenerator): Image data generator for testing data.
        train_images (DirectoryIterator): Iterator for training images.
        val_images (DirectoryIterator): Iterator for validation images.
        test_images (DirectoryIterator): Iterator for testing images.
    """

    # Load the Images with a generator and Data Augmentation
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        **Config.augmentation_args
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator()

    # Flow from DataFrame for training images
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col="Filepath",
        y_col="Label",
        # subset='training',
        **Config.generator_args,
    )

    # Flow from DataFrame for validation images
    val_images = test_generator.flow_from_dataframe(
        dataframe=val_df,
        x_col="Filepath",
        y_col="Label",
        shuffle=False,
        # subset='validation',
        **Config.generator_args,
    )

    # Flow from DataFrame for test images
    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col="Filepath",
        y_col="Label",
        shuffle=False,
        **Config.generator_args,
    )

    return train_generator, test_generator, train_images, val_images, test_images
