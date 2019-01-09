#!/usr/bin/env python3

import os
import numpy as np
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
import data_prep_util
import indoor3d_util

from pathlib import Path

NUM_POINT = 4096
H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 9]
label_dim = [NUM_POINT]
data_dtype = 'float32'
label_dtype = 'uint8'

# Set paths
filelist = os.path.join(BASE_DIR, 'shapes/all_data_label.txt')
cwd = Path(os.path.abspath(os.path.dirname(__file__)))
data = cwd / 'data'
scenes = data / 'scenes'
numpy_dir = data / 'numpy'

data_label_files = [os.path.join(numpy_dir, line.rstrip()) for line in open(filelist)]
cwd = Path(os.path.abspath(os.path.dirname(__file__)))
pwd = cwd.parent
#data = cwd / 'data'
data = pwd
hdf5_data = data / 'hdf5_data'
all_files = hdf5_data / 'all_files.txt'
output_dir = hdf5_data.as_posix()

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_filename_prefix = os.path.join(output_dir, 'ply_data_all')
test_dir = os.path.join(hdf5_data.as_posix(), 'test')
train_dir = os.path.join(hdf5_data.as_posix(), 'train')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
output_room_filelist_train = os.path.join(train_dir, 'room_filelist.txt')
output_room_filelist_test = os.path.join(test_dir, 'room_filelist.txt')
fout_room = open(output_room_filelist, 'w')
fout_room_train = open(output_room_filelist_train, 'w')
fout_room_test = open(output_room_filelist_test, 'w')

# --------------------------------------
# ----- BATCH WRITE TO HDF5 -----
# --------------------------------------
batch_data_dim = [H5_BATCH_SIZE] + data_dim
batch_label_dim = [H5_BATCH_SIZE] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0 # state: the next h5 file to save

def write_to_all_files(all_files, h5_filename):
    """Update hdf5_data/all_files.txt. This file is then loaded in
    train.py as a reference for where to load the data.
    """
    print(f'write_to_all_files all_files = {all_files}')
#   files = [line.rstrip() for line in open(all_files.as_posix())]
    files = [line.rstrip() for line in open(all_files)]
    if h5_filename not in files:
        files.append(h5_filename)
#       with open(all_files.as_posix(), 'w') as f:
        with open(all_files, 'w') as f:
            f.write('\n'.join(sorted(files)))


def insert_batch(data, label, output_dir, is_last_batch=False):
    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]

#   print(f'data.shape = {data.shape}')

    all_files = os.path.join(output_dir, 'all_files.txt')
    print(f'all_files = {all_files}')

    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size+data_size] = label
        buffer_size += data_size
    else: # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert(capacity>=0)
        if capacity > 0:
           h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...]
           h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...]
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename = output_filename_prefix + '_' + str(h5_index) + '.h5'
        h5_filename = os.path.join(output_dir, f'ply_data_{h5_index!s}.h5')
        print(f'h5_filename = {h5_filename}')
        data_prep_util.save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype)
        write_to_all_files(all_files, h5_filename)

        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], output_dir, is_last_batch)
    if is_last_batch and buffer_size > 0:
        h5_filename = output_filename_prefix + '_' + str(h5_index) + '.h5'
        h5_filename = os.path.join(output_dir, f'ply_data_{h5_index!s}.h5')
        print(f'is last batch')
        print(f'h5_filename = {h5_filename}')
        data_prep_util.save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        write_to_all_files(all_files, h5_filename)

        h5_index += 1
        buffer_size = 0

    return


def convert_to_h5(filelist):
    sample_cnt = 0
    data_label_files = [os.path.join(numpy_dir, line.rstrip()) for line in open(filelist)]
    print(f'data_label_files = {data_label_files}')
    for i, data_label_filename in enumerate(data_label_files):
        print(f'data_label_filename = {Path(data_label_filename).name}')
        # Normalize the data here?
        data, label = indoor3d_util.room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=1.0, stride=0.5,
                                                                   random_sample=False, sample_num=None)

        test_or_train_dir = 'test' if 'test' in Path(data_label_filename).name else 'train'

        for _ in range(data.shape[0]):
            if test_or_train_dir == 'test':
                fout_room_test.write(os.path.basename(data_label_filename)[0:-4]+'\n')
            elif test_or_train_dir == 'train':
                fout_room_train.write(os.path.basename(data_label_filename)[0:-4]+'\n')

        sample_cnt += data.shape[0]
        # Need to update is_last_batch if using test & train directories
        is_last_batch = i == len(data_label_files) - 1
        output_dir_test_or_train = os.path.join(hdf5_data.as_posix(), test_or_train_dir)
        insert_batch(data, label, output_dir_test_or_train, is_last_batch)
#       insert_batch(data, label, output_dir, is_last_batch)

    fout_room.close()
    print("Total samples: {0}".format(sample_cnt))


def convert_to_h5_test():
    sample_cnt = 0
    filelist = os.path.join(BASE_DIR, 'shapes/all_data_label_test.txt')
    data_label_files = [os.path.join(numpy_dir, line.rstrip()) for line in open(filelist)]
    for i, data_label_filename in enumerate(data_label_files):
        print(f'data_label_filename = {Path(data_label_filename).name}')
        # Normalize the data here?
        data, label = indoor3d_util.room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=1.0, stride=0.5,
                                                                   random_sample=False, sample_num=None)

        test_or_train_dir = 'test' if 'test' in Path(data_label_filename).name else 'train'

        for _ in range(data.shape[0]):
            if test_or_train_dir == 'test':
                fout_room_test.write(os.path.basename(data_label_filename)[0:-4]+'\n')
            elif test_or_train_dir == 'train':
                fout_room_train.write(os.path.basename(data_label_filename)[0:-4]+'\n')

        sample_cnt += data.shape[0]
        # Need to update is_last_batch if using test & train directories
        is_last_batch = i == len(data_label_files) - 1
        output_dir_test_or_train = os.path.join(hdf5_data.as_posix(), test_or_train_dir)
        insert_batch(data, label, output_dir_test_or_train, is_last_batch)
#       insert_batch(data, label, output_dir, is_last_batch)

    fout_room.close()
    print("Total samples: {0}".format(sample_cnt))


def main():
    filelist = os.path.join(BASE_DIR, 'shapes/all_data_label_test.txt')
    convert_to_h5(filelist)
    filelist = os.path.join(BASE_DIR, 'shapes/all_data_label_train.txt')
    convert_to_h5(filelist)


if __name__ == '__main__':
    main()
