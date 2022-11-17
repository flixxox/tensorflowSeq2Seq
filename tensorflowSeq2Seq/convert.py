
import os
import argparse

import tensorflow as tf

from tensorflowSeq2Seq.util.globals import Globals
from tensorflowSeq2Seq.util.debug import my_print
from tensorflowSeq2Seq.util.config import Config
from tensorflowSeq2Seq.model.models import Model
from tensorflowSeq2Seq.util.data import (Vocabulary, Dataset)
from tensorflowSeq2Seq.search.search_algorithm_selector import SearchAlgorithmSelector


def parse_cli_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)

    return vars(parser.parse_args())


def convert(config):

    epoch = tf.Variable(1)

    vocab_src = Vocabulary.create_vocab(config['vocab_src'])
    vocab_tgt = Vocabulary.create_vocab(config['vocab_tgt'])

    saved_model_dir = os.path.join(config['output_folder'], 'saved_model')
    tflite_dir = os.path.join(config['output_folder'], 'tflite')
    
    model = Model.create_model_from_config(config,
        vocab_src.vocab_size,
        vocab_tgt.vocab_size,
    )

    search_dataset = Dataset.create_dataset_from_config(config['quant_src'], config['quant_tgt'], vocab_src, vocab_tgt)
    tf_dataset = search_dataset.get_prepared_tf_dataset(config['batch_size'])

    search_algorithm = SearchAlgorithmSelector.create_search_algorithm_from_config(config, model, vocab_src, vocab_tgt)

    checkpoint = tf.train.Checkpoint(epoch=epoch, model=model)

    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, 
        config['model_folder'],
        max_to_keep=None
    )

    my_print('Restoring checkpoint!')
    checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
    my_print(f'Restoring epoch {epoch.numpy()}!')


    # SavedModel: Save 
    tf.saved_model.save(
        model, saved_model_dir,
        signatures=search_algorithm.search_batch.get_concrete_function(
            src = tf.TensorSpec(shape=[None, None], dtype=tf.int32)
        )
    )

    # Representative dataset for quantization
    def representative_dataset():
        for src, _, _, _ in tf_dataset:
            yield {
                "src": src
        }

    # TFLite: Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]
    # converter.representative_dataset = representative_dataset
    # converter.inference_input_type = tf.int8  # or tf.uint8
    # converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_model = converter.convert()

    # TFLite: Save the model.
    with open(os.path.join(tflite_dir, 'model.tflite'), 'wb') as f:
        f.write(tflite_model)
    
    my_print('Done!')


if __name__ == '__main__':

    my_print(''.center(40, '-'))
    my_print(' Hi! '.center(40, '-'))
    my_print(' Script: convert.py '.center(40, '-'))
    my_print(''.center(40, '-'))

    args = parse_cli_arguments()

    config = Config.parse_config(args)
    Globals.set_train_flag(False)

    config.print_config()

    convert(config)