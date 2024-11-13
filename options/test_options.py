from .base_options import BaseOptions
import oyaml as yaml
import argparse
import sys




class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--train_config', type=argparse.FileType(mode='r'), required=True, help='config file saved from model training')
        parser.add_argument('--partition', type=str, default='val', help='val or test')
        parser.add_argument('--dataset_name', type=str, required=True, help="name to describe test dataset when saving results, e.g. celebahq_pgan")
        parser.add_argument('--force_redo', action='store_true', help="force recompute results")

        # for testing model robustness (additional augmentations)
        parser.add_argument('--test_compression', type=int, help='jpeg compression level')
        parser.add_argument('--test_gamma', type=int, help='gamma adjustment level')
        parser.add_argument('--test_blur', type=int, help='blur level')
        parser.add_argument('--test_flip', action='store_true', help='flip all test images')

        # visualizations
        parser.add_argument('--visualize', action='store_true', help='save visualizations when running test')
        parser.add_argument('--average_mode', help='which kind of patch averaging to use for visualizations [vote, before_softmax, after_softmax]')
        parser.add_argument('--topn', type=int, default=100, help='visualize top n')
        return parser