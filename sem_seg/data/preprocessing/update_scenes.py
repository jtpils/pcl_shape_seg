#!/usr/bin/env python3

import argparse
import os
import shutil

from pathlib import Path

cwd = Path(os.path.abspath(os.path.dirname(__file__)))
data = cwd.parent
scenes = data / 'scenes'
numpy_dir = data / 'numpy'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str)
    flags = parser.parse_args()
    return flags


def update_scene(scene, test_or_train):
    filename = scene / (scene.stem + '.txt')
    annotation = scene / 'annotation'

    anno_path = annotation.relative_to(data.as_posix())
    anno_paths = [line.rstrip() for line in open(f'shapes/anno_shapes_{test_or_train}.txt')]
    if anno_path.as_posix() not in anno_paths:
        anno_paths.append(anno_path.as_posix())
        with open(f'shapes/anno_shapes_{test_or_train}.txt', 'w') as f:
            f.write('\n'.join(sorted(anno_paths)))

    test_or_train = 'test' if 'test' in scene.as_posix() else 'train'
    npy_file = numpy_dir / test_or_train / ((test_or_train + '_') + (scene.stem + '.npy'))
    all_data_label = [line.rstrip() for line in open(f'shapes/all_data_label_{test_or_train}.txt')]
    if npy_file.as_posix() not in all_data_label:
        all_data_label.append(npy_file.as_posix())
        with open(f'shapes/all_data_label_{test_or_train}.txt', 'w') as f:
            f.write('\n'.join(sorted(all_data_label)))


def main():
    args = parse_args()
    scene = Path(args.scene)
    update_scene(scene)


if __name__ == '__main__':
    main()
