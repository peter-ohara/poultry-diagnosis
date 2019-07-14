# Copyright 2014-2017 Bert Carremans
# Author: Bert Carremans <bertcarremans.be>
#
# License: BSD 3 clause

import os
from random import randrange
from shutil import copyfile


def img_train_test_split(img_source_dir, test_size, train_dir='train', test_dir='test'):
    """
    Randomly splits images over a train, and test folder, while preserving the folder structure
    
    Parameters
    ----------
    img_source_dir : string
        Path to the folder with the images to be split. Can be absolute or relative path

    test_size : float
        Proportion of the original images that need to be copied in the subdirectory in the test folder
    """
    if not (isinstance(img_source_dir, str)):
        raise AttributeError('img_source_dir must be a string')

    if not os.path.exists(img_source_dir):
        raise OSError('img_source_dir does not exist')

    if not (isinstance(test_size, float)):
        raise AttributeError('test_size must be a float')

    train_dir_path = f'data/{train_dir}'
    test_dir_path = f'data/{test_dir}'

    setup_empty_folder_structure(test_dir_path, train_dir_path)

    # Get the subdirectories in the main image folder
    subdirs = [subdir for subdir in os.listdir(img_source_dir) if os.path.isdir(os.path.join(img_source_dir, subdir))]

    copy_files_by_test_size(img_source_dir, subdirs, test_dir_path, test_size, train_dir_path)


def copy_files_by_test_size(img_source_dir, subdirs, test_dir_path, test_size, train_dir_path):
    for subdir in subdirs:
        subdir_fullpath = os.path.join(img_source_dir, subdir)
        files = os.listdir(subdir_fullpath)

        num_files = len(files)
        if num_files == 0:
            print(f'{subdir_fullpath} is empty')
            break
        else:
            print(f'{num_files} images in {subdir}')

        # Randomly assign an image to train or test folder
        train, test = train_test_split(files, split=test_size)

        copy_files(subdir, subdir_fullpath, train, train_dir_path)
        copy_files(subdir, subdir_fullpath, test, test_dir_path)

        print()


def setup_empty_folder_structure(test_dir_path, train_dir_path):
    # Set up empty folder structure if not exists
    if not os.path.exists('data'):
        os.makedirs('data')
    else:
        if not os.path.exists(train_dir_path):
            os.makedirs(train_dir_path)
        if not os.path.exists(test_dir_path):
            os.makedirs(test_dir_path)


# Split a dataset into a train and test set
def train_test_split(dataset, split=0.30):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


def copy_files(subdir, subdir_fullpath, train, train_dir_path):
    # Create subdirectory in train folder
    train_subdir = os.path.join(train_dir_path, subdir)
    if not os.path.exists(train_subdir):
        os.makedirs(train_subdir)
    # Copy train files
    counter = 0
    for filename in train:
        basename = os.path.basename(filename)
        copyfile(os.path.join(subdir_fullpath, filename), os.path.join(train_subdir, basename))
        counter += 1

    print(f'Copied {str(counter)} images to {train_dir_path}/{subdir}')
