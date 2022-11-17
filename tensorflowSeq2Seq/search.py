
import argparse
from os.path import join

import tensorflow as tf
import horovod.tensorflow as hvd

from models import create_model_from_config
from tensorflowSeq2Seq.util.config import Config
from tensorflowSeq2Seq.util.globals import Globals
from tensorflowSeq2Seq.util.setup import setup_tf_from_config
from tensorflowSeq2Seq.util.checkpoint_manager import CheckpointManager
from tensorflowSeq2Seq.util.debug import my_print, get_number_of_trainable_variables
from tensorflowSeq2Seq.search.search_algorithm_selector import SearchAlgorithmSelector
from tensorflowSeq2Seq.util.data import Vocabulary, Dataset, BatchGenerator, LinearBatchAlgorithm


def parse_cli_arguments():

    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, 
        help='The path to the config.yaml which contains all user defined parameters. It may or may not match the one trained with. This is up to the user to ensure.')
    parser.add_argument('--checkpoint-prefix', type=str, required=True, 
        help='The checkpoint prefix pointing to the model weights.')
    parser.add_argument('--output-folder', type=str, required=False, default=None, 
        help='The output folder in which to write the score and hypotheses.')
    parser.add_argument('--number-of-gpus', type=int, required=False, default=None, 
        help='This is usually specified in the config but can also be overwritten from the cli. However, in search this can only be 0 or 1. We do not support multi-gpu decoding. If you set it to >1 we will set it back to 1 so that you dont need to modify the config in search.')

    args = parser.parse_args()

    return vars(args)


def search(config):

    setup_tf_from_config(config)

    epoch = tf.Variable(1)

    config['batch_size'] = config['batch_size_search']

    vocab_src = Vocabulary.create_vocab(config['vocab_src'])
    vocab_tgt = Vocabulary.create_vocab(config['vocab_tgt'])

    my_print('Vocab Size Src', vocab_src.vocab_size)
    my_print('Vocab Size Tgt', vocab_tgt.vocab_size)

    search_dataset = Dataset.create_dataset_from_config(config, 'dev_set', config['dev_src'], config['dev_tgt'], vocab_src, vocab_tgt)
    
    search_batch_generator = BatchGenerator.create_batch_generator_from_config(config, search_dataset, LinearBatchAlgorithm)

    if config['threaded_data_loading']:
        search_batch_generator.start()

    model = create_model_from_config(config, vocab_src, vocab_tgt)

    my_print(f'Trainable variables: {get_number_of_trainable_variables(model)}')

    search_algorithm = SearchAlgorithmSelector.create_search_algorithm_from_config(config, model, vocab_src, vocab_tgt)

    checkpoint_manager = CheckpointManager.create_eval_checkpoint_manager_from_config(config, epoch, model)
    checkpoint_manager.restore(config['checkpoint_prefix'], partial=True)

    if config['checkpoint_prefix'].endswith('best'):
        output_file = join(config['output_folder'], 'hyps_best')
    else:
        output_file = join(config['output_folder'], f'hyps_{epoch.numpy()}')

    my_print(f'Searching epoch {epoch.numpy()}!')

    search_algorithm.search(search_batch_generator, output_file)

    if config['threaded_data_loading']:
        search_batch_generator.stop()

    my_print('Done!')


if __name__ == '__main__':

    hvd.init()

    my_print(''.center(40, '-'))
    my_print(' Hi! '.center(40, '-'))
    my_print(' Script: search.py '.center(40, '-'))
    my_print(''.center(40, '-'))

    args = parse_cli_arguments()

    config = Config.parse_config(args)
    Globals.set_train_flag(False)
    Globals.set_time_flag(False)

    config.print_config()

    search(config)