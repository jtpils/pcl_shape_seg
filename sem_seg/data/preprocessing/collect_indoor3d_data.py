#!/usr/bin/env python3

import os
import sys

from pathlib import Path

cwd = Path(os.path.abspath(os.path.dirname(__file__)))
pwd = cwd.parent
sys.path.append(cwd.as_posix())
import indoor3d_util


def write_annotation_to_numpy(annotation_file):
#   data = cwd / 'data'
    data = pwd
    numpy_dir = data / 'numpy'
    filename = (Path(cwd) / Path(annotation_file)).as_posix()
    annotations = [line.rstrip() for line in open(filename)]
    annotations = [os.path.join(data, p) for p in annotations]

    for annotation in annotations:
        elements = annotation.split('/')
        test_or_train = 'test' if 'test' in elements else 'train'
        filename = test_or_train + '_' + ''.join([elements[-2], '.npy'])
        outfile = os.path.join(numpy_dir, elements[-4], filename)
        print(f'Writing to {outfile}')
        try:
            indoor3d_util.collect_point_label(annotation, outfile, 'numpy')
        except:
            print(f'Failed to write {outfile}')

def main():
    write_annotation_to_numpy('shapes/anno_shapes_train.txt')
    write_annotation_to_numpy('shapes/anno_shapes_test.txt')


if __name__ == '__main__':
    main()
