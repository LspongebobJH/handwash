#!/usr/bin/env python
import argparse
import sys

# torchlight
import torchlight
from torchlight import import_class
import os

import numpy as np
import torch
import random

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # Set random seed for the built-in random module
    random.seed(42)

    # Set random seed for numpy
    np.random.seed(42)

    # Set random seed for PyTorch
    torch.manual_seed(42)

    # If you're using CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    processors['demo_old'] = import_class('processor.demo_old.Demo')
    processors['demo'] = import_class('processor.demo_realtime.DemoRealtime')
    processors['demo_offline'] = import_class('processor.demo_offline.DemoOffline')
    #endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # start

    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    p.start()
