#!/usr/bin/env python3

import os

from pathlib import Path

from update_scenes import update_scene


cwd = Path(os.path.abspath(os.path.dirname(__file__)))
data = cwd.parent
scenes = data / 'scenes'


def main():
    for test_or_train in ['test', 'train']:
        for distribution in ['uniform', 'blender']:
            scene_dir = scenes / test_or_train / distribution
            if scene_dir.is_dir():
                for scene in scene_dir.iterdir():
                    print(f'Preparing scene {scene}')
                    update_scene(scene, test_or_train)


if __name__ == '__main__':
    main()
