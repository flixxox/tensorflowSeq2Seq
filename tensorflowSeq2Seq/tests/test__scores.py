import unittest

import tensorflow as tf
import horovod.tensorflow as hvd
from numpy.testing import assert_allclose

from tensorflowSeq2Seq.util.config import Config
from tensorflowSeq2Seq.util.debug import my_print
from tensorflowSeq2Seq.model.scores import BatchScore

class TestScores(unittest.TestCase):


    def test_batchscores__no_memory(self):

        hvd.init()

        p = 0

        tgt = [
            [p, p, p],
            [p, p, p],
            [p, p, p],
        ]

        self.__apply_score_and_check(tgt, 19, 0.0, 1.0)

        tgt = [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ]

        self.__apply_score_and_check(tgt, 9, 1.0, 0.0)
        self.__apply_score_and_check(tgt, 18, 0.5, 0.0)

        tgt = [
            [1, 2, 0, 0, 0],
            [1, 2, 0, 0, 0],
            [1, 2, 3, 4, 5],
        ]

        self.__apply_score_and_check(tgt, 10, 0.9, (6/15))
    
    def __apply_score_and_check(self, tgt, batch_size, fullness_expected, redundancy_expected, p=0):

        config = Config(
            {
                'batch_size': batch_size,
                'max_sentence_length': 128,
            }
        )

        batch_score = BatchScore.create_score_from_config(config, p)

        tgt = tf.constant(tgt, dtype=tf.int32)

        batch_score(tgt)

        fullness, redundancy, _ = batch_score.average_and_reset()

        assert_allclose(fullness, fullness_expected, rtol=1e-06, atol=0, err_msg=f'Fullness expected {fullness_expected}, Got {fullness}')
        assert_allclose(redundancy, redundancy_expected, rtol=1e-06, atol=0, err_msg=f'Redundancy expected {redundancy_expected}, Got {redundancy}')


    def test_batchscores__memory(self):

        hvd.init()

        config = Config(
            {
                'batch_size': 16,
                'max_sentence_length': 128,
            }
        )

        p = 0

        batch_score = BatchScore.create_score_from_config(config, p)

        tgt = [
            [p, 1, p],
            [4, p, 2],
            [p, 3, p],
        ]

        tgt = tf.constant(tgt, dtype=tf.int32)

        batch_score(tgt)

        tgt = [
            [p,p,p,p,p],
            [p,p,p,p,p],
            [p,p,p,p,p]
        ]

        tgt = tf.constant(tgt, dtype=tf.int32)

        batch_score(tgt)

        fullness, redundancy, _ = batch_score.average_and_reset()

        fullness_expected = ((4/16) + (0/16)) / 2
        redundancy_expected = ((5/9) + (15/15)) / 2 

        assert_allclose(fullness, fullness_expected, rtol=1e-06, atol=0, err_msg=f'Fullness expected {fullness_expected}, Got {fullness}')
        assert_allclose(redundancy, redundancy_expected, rtol=1e-06, atol=0, err_msg=f'Redundancy expected {redundancy_expected}, Got {redundancy}')