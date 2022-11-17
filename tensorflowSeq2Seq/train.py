
import time
import argparse
from os.path import join

import tensorflow as tf
import horovod.tensorflow as hvd

from models import create_model_from_config
from tensorflowSeq2Seq.util.config import Config
from tensorflowSeq2Seq.model.scores import Score
from tensorflowSeq2Seq.util.globals import Globals
from tensorflowSeq2Seq.util.trainer import Trainer
from tensorflowSeq2Seq.model.optimizers import Optimizer
from tensorflowSeq2Seq.util.setup import setup_tf_from_config
from tensorflowSeq2Seq.util.checkpoint_manager import CheckpointManager
from tensorflowSeq2Seq.util.debug import my_print, get_number_of_trainable_variables, tf_print_memory_usage
from tensorflowSeq2Seq.util.data import Vocabulary, Dataset, BatchGenerator, BucketingBatchAlgorithm, LinearBatchAlgorithm

def parse_cli_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, 
        help='The path to the config.yaml which contains all user defined parameters.')
    parser.add_argument('--output-folder', type=str, required=True, 
        help='The folder in which to write the training output (ckpts, learning-rates, perplexities etc.)')
    parser.add_argument('--resume-training', type=int, required=False, default=False, 
        help='If you want to resume a training, set this flag to 1 and specify the directory with "resume-training-from".')
    parser.add_argument('--resume-training-from', type=str, required=False, default='', 
        help='If you want to resume a training, specify the output directory here. We expect it to have the same layout as a newly created one.')
    parser.add_argument('--number-of-gpus', type=int, required=False, default=None,
        help='This is usually specified in the config but can also be overwritten from the cli.')

    args = parser.parse_args()

    args.resume_training = bool(args.resume_training)

    return vars(args)


def train(config):

    setup_tf_from_config(config)
        
    num_epochs      = config['epochs']
    numbers_dir     = join(config['output_folder'], 'numbers')
    epoch           = tf.Variable(0)

    vocab_src = Vocabulary.create_vocab(config['vocab_src'])
    vocab_tgt = Vocabulary.create_vocab(config['vocab_tgt'])

    my_print('Vocab Size Src', vocab_src.vocab_size)
    my_print('Vocab Size Tgt', vocab_tgt.vocab_size)
    
    train_dataset   = Dataset.create_dataset_from_config(config, 'train_set',   config['train_src'],    config['train_tgt'],    vocab_src, vocab_tgt)
    dev_dataset     = Dataset.create_dataset_from_config(config, 'dev_set',     config['dev_src'],      config['dev_tgt'],      vocab_src, vocab_tgt)

    train_batch_generator   = BatchGenerator.create_batch_generator_from_config(config, train_dataset,  BucketingBatchAlgorithm, chunking=config['update_freq'])
    dev_batch_generator     = BatchGenerator.create_batch_generator_from_config(config, dev_dataset,    LinearBatchAlgorithm)

    if config['threaded_data_loading']:
        train_batch_generator.start()
        dev_batch_generator.start()
    

    model = create_model_from_config(config, vocab_src, vocab_tgt)

    my_print(f'Trainable variables: {get_number_of_trainable_variables(model)}')

    criterion   = Score.create_score_from_config(config)
    optimizer   = Optimizer.create_optimizer_from_config(config, numbers_dir)
    trainer     = Trainer.create_trainer_from_config(config, train_batch_generator, dev_batch_generator, model, criterion, optimizer)
    
    checkpoint_manager = CheckpointManager.create_train_checkpoint_manager_from_config(config, epoch, optimizer, model)
    checkpoint_manager.restore_or_initialize(model, seed=config['seed'])

    epoch = epoch.assign_add(1)

    accum_epoch_times   = 0
    epochs_trained      = 0

    tf_print_memory_usage()
    my_print(f'Start training at epoch {epoch.numpy()}!')

    for e in range(int(epoch.numpy()), num_epochs+1):

        epoch = epoch.assign(e)

        start = time.perf_counter()

        train_ppl   = trainer.train(e)
        dev_ppl     = trainer.eval(e)

        end = time.perf_counter()
        my_print(f'Epoch took: {end - start:4.2f}s, {(end - start) / 60:4.2f}min')

        tf_print_memory_usage()

        checkpoint_manager.save(epoch, dev_ppl)

        Score.write_score_to_file(numbers_dir, 'train_ppl', train_ppl)
        Score.write_score_to_file(numbers_dir, 'dev_ppl',   dev_ppl)

        accum_epoch_times   += end - start
        epochs_trained      += 1

        if checkpoint_manager.early_abort():
            my_print('Early aborting!')
            break

    if config['threaded_data_loading']:
        train_batch_generator.stop()
        dev_batch_generator.stop()

    my_print(f'Average time per epoch: {accum_epoch_times / epochs_trained:4.2f}s {(accum_epoch_times / epochs_trained) / 60:4.2f}min')
    my_print('Done!')


if __name__ == '__main__':

    hvd.init()

    my_print(''.center(40, '-'))
    my_print(' Hi! '.center(40, '-'))
    my_print(' Script: train.py '.center(40, '-'))
    my_print(''.center(40, '-'))

    args = parse_cli_arguments()

    config = Config.parse_config(args)
    Globals.set_train_flag(True)
    Globals.set_time_flag(False)

    config.print_config()

    train(config)