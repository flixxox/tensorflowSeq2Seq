import unittest

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflowSeq2Seq.util.config import Config
from tensorflowSeq2Seq.util.debug import my_print
from tensorflowSeq2Seq.util.globals import Globals
from tensorflowSeq2Seq.util.data import Vocabulary, Dataset, BatchGenerator, BucketingBatchAlgorithm, LinearBatchAlgorithm

class TestData(unittest.TestCase):

    def test_bucketing_algorithm__in_memory__no_threading(self):

        self.__test_data_reading(True, False, BucketingBatchAlgorithm)

    def test_bucketing_algorithm__in_memory__threading(self):

        self.__test_data_reading(True, True, BucketingBatchAlgorithm)
    
    def test_linear_algorithm__in_memory__no_threading(self):

        self.__test_data_reading(True, False, LinearBatchAlgorithm)
    
    def test_linear_algorithm__in_memory__threading(self):

        self.__test_data_reading(True, True, LinearBatchAlgorithm)


    def __test_data_reading(self, in_memory, threading, algorithm):

        hvd.init()

        path_test_vocab_src = 'tests/res/mockup_vocab_src.pickle'
        path_test_vocab_tgt = 'tests/res/mockup_vocab_tgt.pickle'
        path_test_data_src  = 'tests/res/mockup_data_src'
        path_test_data_tgt  = 'tests/res/mockup_data_tgt'

        max_sentence_length     = 256
        batch_size              = 100
        threaded_data_loading   = False
        epoch_split             = 4

        data_src, data_tgt                              = self.__read_test_data(path_test_data_src, path_test_data_tgt)
        vocab_src, vocab_tgt, dataset, batch_generator  = self.__prepare_testable_objects(algorithm, path_test_vocab_src, path_test_vocab_tgt, path_test_data_src, path_test_data_tgt, batch_size, max_sentence_length, epoch_split, in_memory, threaded_data_loading)
        pad_index                                       = vocab_src.PAD

        if threading:
            batch_generator.start()

        assert pad_index == vocab_tgt.PAD

        data_read_src, data_read_tgt = self.__generate_batches(batch_generator, vocab_src, vocab_tgt, batch_size, epoch_split, pad_index)

        self.__check_read_against_file_shuffled(data_src.copy(), data_read_src, data_tgt.copy(), data_read_tgt)

        if threading:
            batch_generator.stop()

    def __read_test_data(self, test_data_src, test_data_tgt):

        src_data = []
        tgt_data = []

        with open(test_data_src, "r") as src_file, open(test_data_tgt, "r") as tgt_file:
            
            for (src, tgt) in zip(src_file, tgt_file):

                src = src.strip().replace("\n", "").split(" ")
                tgt = tgt.strip().replace("\n", "").split(" ")

                src_data.append(src)
                tgt_data.append(tgt)

        return src_data, tgt_data

    def __prepare_testable_objects(self, algorithm_class, path_test_vocab_src, path_test_vocab_tgt, path_test_data_src, path_test_data_tgt, batch_size, max_sentence_length, epoch_split, in_memory, threaded_data_loading):

        Globals.set_number_of_workers(1)
        Globals.set_train_flag(True)

        config = Config(
            {
                'dataset':                  'TranslationDataset',
                'batch_size':               batch_size,
                'max_sentence_length':      max_sentence_length,
                'threaded_data_loading':    threaded_data_loading,
                'load_datset_in_memory':    in_memory,
                'epoch_split':              epoch_split
            }
        )

        vocab_src = Vocabulary.create_vocab(path_test_vocab_src)
        vocab_tgt = Vocabulary.create_vocab(path_test_vocab_tgt)

        dataset = Dataset.create_dataset_from_config(config, 'test_set', path_test_data_src, path_test_data_tgt, vocab_src, vocab_tgt)
        
        batch_generator = BatchGenerator.create_batch_generator_from_config(config, dataset, algorithm_class)

        return vocab_src, vocab_tgt, dataset, batch_generator

    def __generate_batches(self, batch_generator, vocab_src, vocab_tgt, batch_size, epoch_split, pad_index):

        data_read_src = []
        data_read_tgt = []

        for i in range(epoch_split):
            for _, (src, tgt, out) in batch_generator.generate_batches():

                assert src.shape[0] == tgt.shape[0] == out.shape[0]

                assert tgt.shape[1] == out.shape[1]

                self.__check_batch_size(tgt, batch_size, pad_index)

                src = list(src.numpy())
                tgt = list(tgt.numpy())

                for s, t in zip(src, tgt):
                    
                    s = list(s)
                    t = list(t)

                    s = vocab_src.detokenize(s)
                    t = vocab_tgt.detokenize(t)

                    s = vocab_src.remove_padding(s)
                    t = vocab_tgt.remove_padding(t)

                    s = s[1:-1]
                    t = t[1:]

                    data_read_src.append(s)
                    data_read_tgt.append(t)

        return data_read_src, data_read_tgt

    def __check_batch_size(self, tgt, batch_size, pad_index):

        trainable_tokens_in_batch = tf.cast(tgt != pad_index, dtype=tf.int32)
        trainable_tokens_in_batch = tf.reduce_sum(trainable_tokens_in_batch).numpy()

        assert trainable_tokens_in_batch <= batch_size or tgt.shape[0] == 1

    def __check_read_against_file_in_order(self, data_src, data_read_src, data_tgt, data_read_tgt):

        assert len(data_src) == len(data_read_src) == len(data_tgt) == len(data_read_tgt)

        for src, src_read, tgt, tgt_read in zip(data_src, data_read_src, data_tgt, data_read_tgt):

            assert len(src) == len(src_read), f'src {src}, src_read {src_read}'
            assert len(tgt) == len(tgt_read), f'src {tgt}, src_read {tgt_read}'

            for i in range(len(src)):
                assert src[i] == src_read[i]
            
            for i in range(len(tgt)):
                assert tgt[i] == tgt_read[i]

    def __check_read_against_file_shuffled(self, data_src, data_read_src, data_tgt, data_read_tgt):

        assert len(data_src) == len(data_read_src) == len(data_tgt) == len(data_read_tgt)

        for src_read, tgt_read in zip(data_read_src, data_read_tgt):

            assert src_read in data_src
            assert tgt_read in data_tgt

            data_src.remove(src_read)
            data_tgt.remove(tgt_read)

        assert len(data_src) == len(data_tgt) == 0
