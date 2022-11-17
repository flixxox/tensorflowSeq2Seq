
import os
import sys
import argparse

import tensorflow as tf

import _setup_env
from tensorflowSeq2Seq.util.config import Config
from tensorflowSeq2Seq.model.models import Model
from tensorflowSeq2Seq.search.search_algorithm_selector import SearchAlgorithmSelector
from tensorflowSeq2Seq.util.data import (Vocabulary, Dataset)


def parse_cli_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--tflite-model-path', type=str, required=True)

    return vars(parser.parse_args())


def test(config):

    vocab_src = Vocabulary.create_vocab(config['vocab_src'])
    vocab_tgt = Vocabulary.create_vocab(config['vocab_tgt'])

    search_dataset = Dataset.create_dataset_from_config(config['dev_src'], config['dev_tgt'], vocab_src, vocab_tgt)

    search_algorithm = SearchAlgorithmSelector.create_search_algorithm_from_config(config, None, vocab_src, vocab_tgt)

    interpreter = tf.lite.Interpreter(model_path=config['tflite_model_path'])

    search_algorithm.search_tflite(interpreter, search_dataset)

    print('Done!')


if __name__ == '__main__':

    print(''.center(40, '-'))
    print(' Hi! '.center(40, '-'))
    print(' Script: test_tflite_model.py '.center(40, '-'))
    print(''.center(40, '-'))

    args = parse_cli_arguments()

    config = Config.parse_config(args)

    config.print_config()

    test(config)