"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras_retinanet.bin.train

import warnings
import sys


def test_coco():
    # ignore warnings in this test
    warnings.simplefilter('ignore')

    # simulate arguments
    sys.argv = [
        '',
        '--epochs=1',
        '--steps=1',
        '--no-snapshots',
        'coco',
        'tests/test-data/coco',
    ]

    # run training / evaluation
    keras_retinanet.bin.train.main()