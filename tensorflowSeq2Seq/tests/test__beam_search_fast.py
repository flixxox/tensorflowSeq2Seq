import unittest

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflowSeq2Seq.util.config import Config
from tensorflowSeq2Seq.model.models import Model
from tensorflowSeq2Seq.util.debug import my_print
from tensorflowSeq2Seq.util.data import Vocabulary
from tensorflowSeq2Seq.search.beam_search_fast import BeamSearchFast

class TestData(unittest.TestCase):

    def test_get_active_model_input(self):
        
        B = 3
        srcT = 6
        i = 5
        N = 4
        V = 10
        maxI = 3
        EOS = 7
        PAD = -1

        beam_search = self.__setup_test_object(B, maxI, N, V, EOS, PAD)

        src = tf.range(B*N*srcT, dtype=tf.int32)
        src = tf.reshape(src, (B, N, srcT))

        tgt = tf.range(B*N*i, dtype=tf.int32)
        tgt = tf.reshape(tgt, (B, N, i))

        active_mask = tf.constant([
            [
                True, False, True, False
            ],
            [
                False, False, False, False
            ],
            [
                False, True, False, True
            ],
        ], dtype=tf.bool)
        BNa = 4

        print(src)
        print(tgt)

        tf.debugging.assert_equal(tf.shape(src),            [B, N, srcT])
        tf.debugging.assert_equal(tf.shape(tgt),            [B, N, i])
        tf.debugging.assert_equal(tf.shape(active_mask),    [B, N])

        exp_srca = tf.constant([
            [ 0,  1,  2,  3,  4,  5],
            [12, 13, 14, 15, 16, 17],
            [54, 55, 56, 57, 58, 59],
            [66, 67, 68, 69, 70, 71],
        ], dtype=tf.int32)

        exp_tgta = tf.constant([
            [ 0,  1,  2,  3,  4],
            [10, 11, 12, 13, 14],
            [45, 46, 47, 48, 49],
            [55, 56, 57, 58, 59],
        ], dtype=tf.int32)

        tf.debugging.assert_equal(tf.shape(exp_srca),            [BNa, srcT])
        tf.debugging.assert_equal(tf.shape(exp_tgta),            [BNa, i])

        srca, tgta = beam_search.get_active_model_input(src, tgt, active_mask)

        srca = srca.numpy()
        tgta = tgta.numpy()

        np.testing.assert_array_equal(srca, exp_srca)
        np.testing.assert_array_equal(tgta, exp_tgta)

    def test_pad_to_N(self):
        
        B = 3
        maxI = 3
        BNa = 4
        N = 4
        V = 4
        EOS = 7

        beam_search = self.__setup_test_object(B, maxI, N, V, EOS, -1)

        precomp_indices = tf.expand_dims(tf.range(N), axis=0)
        precomp_indices = tf.repeat(precomp_indices, B, axis=0) + tf.expand_dims(tf.range(B), axis=1) * N

        active_mask = tf.constant([
            [True, False, True, False],
            [False, True, False, True],
            [False, False, False, False],
        ], dtype=tf.bool)

        output = tf.constant([
                [ 1, 2, 3, 4],
                [ 5, 6, 7, 8],
                [13,14,15,16],
                [17,18,19,20],
            ], dtype=tf.float32)

        tf.debugging.assert_equal(tf.shape(output),             [BNa, V])
        tf.debugging.assert_equal(tf.shape(active_mask),        [B, N])
        tf.debugging.assert_equal(tf.shape(precomp_indices),    [B, N])

        exp_output = tf.constant([
            [
                [ 1, 2, 3, 4],
                [ -float('inf'), -float('inf'), -float('inf'), -float('inf')],
                [ 5, 6, 7, 8],
                [ -float('inf'), -float('inf'), -float('inf'), -float('inf')],
            ],
            [
                [ -float('inf'), -float('inf'), -float('inf'), -float('inf')],
                [13,14,15,16],
                [ -float('inf'), -float('inf'), -float('inf'), -float('inf')],
                [17,18,19,20],
            ],
            [
                [ -float('inf'), -float('inf'), -float('inf'), -float('inf')],
                [ -float('inf'), -float('inf'), -float('inf'), -float('inf')],
                [ -float('inf'), -float('inf'), -float('inf'), -float('inf')],
                [ -float('inf'), -float('inf'), -float('inf'), -float('inf')],
            ],
        ], dtype=tf.float32)

        tf.debugging.assert_equal(tf.shape(exp_output), [B, N, V])

        output = beam_search.pad_to_N(output, active_mask, precomp_indices)

        output = output.numpy()
        exp_output = exp_output.numpy()

        np.testing.assert_array_equal(output, exp_output)
    
    def test_update_tgt(self):
        
        B = 3
        i = 3
        N = 3
        maxI = 7
        V = 10
        EOS = 9
        PAD = -1

        beam_search = self.__setup_test_object(B, maxI, N, V, EOS, PAD)

        tgt = tf.constant([
            [
                [1, 2,   9],
                [4, 5,   6],
                [7, 9, PAD],
            ],
            [
                [10, 11,  12],
                [13, 9,  PAD],
                [16, 17,  18],
            ],
            [
                [18, 19,   9],
                [20, 21,   9],
                [22,  9, PAD],
            ]
        ], dtype=tf.int32)

        active_mask = tf.constant([
            [False, True, False],
            [True, False, True],
            [False, False, False],
        ], dtype=tf.bool)

        best_beams = tf.constant([
            [1, 1, 1],
            [2, 0, 0],
            [0, 0, 0],
        ], dtype=tf.int32)

        best_words = tf.constant([
            [  9, 100, 101],
            [102,   9, 103],
            [  0,   1,   2],
        ], dtype=tf.int32)

        scores = tf.constant([
            [-1, -2, -3],
            [-4, -5, -6],
            [-7, -8, -9],
        ], dtype=tf.float32)

        tf.debugging.assert_equal(tf.shape(tgt),                [B, N, i])
        tf.debugging.assert_equal(tf.shape(active_mask),        [B, N])
        tf.debugging.assert_equal(tf.shape(best_beams),         [B, N])
        tf.debugging.assert_equal(tf.shape(best_words),         [B, N])
        tf.debugging.assert_equal(tf.shape(scores),             [B, N])

        exp_tgt = tf.constant([
            [
                [1, 2,   9, PAD],
                [4, 5,   6,   9],
                [7, 9, PAD, PAD],
            ],
            [
                [16, 17,  18, 102],
                [13, 9,  PAD, PAD],
                [10, 11,  12,   9],
            ],
            [       
                [18, 19,   9, PAD],
                [20, 21,   9, PAD],
                [22,  9, PAD, PAD],
            ]
        ], dtype=tf.int32)

        exp_scores = tf.constant([
            [-float('inf'),            -1, -float('inf')],
            [           -4, -float('inf'),            -5],
            [-float('inf'), -float('inf'), -float('inf')],
        ], dtype=tf.float32)

        exp_best_words = tf.constant([
            [PAD,   9, PAD],
            [102, PAD,   9],
            [PAD, PAD, PAD],
        ], dtype=tf.int32)

        tf.debugging.assert_equal(tf.shape(exp_tgt),        [B, N, i+1])
        tf.debugging.assert_equal(tf.shape(exp_scores),     [B, N])
        tf.debugging.assert_equal(tf.shape(exp_best_words), [B, N])

        tgt, scores, best_words = beam_search.update_tgt(tgt, active_mask, best_beams, best_words, scores)

        tgt     = tgt.numpy()
        scores  = scores.numpy()

        exp_tgt         = exp_tgt.numpy()
        exp_scores      = exp_scores.numpy()
        exp_best_words  = exp_best_words.numpy()

        np.testing.assert_allclose(tgt,     exp_tgt)
        np.testing.assert_allclose(scores,  exp_scores)
        np.testing.assert_allclose(best_words,  exp_best_words)

    def test_update_storage(self):
        
        B = 4
        N = 3
        V = 10
        maxI = 8
        EOS = 9
        PAD = -1

        beam_search = self.__setup_test_object(B, maxI, N, V, EOS, PAD)

        fin_storage_scores = tf.constant([
            [-float('inf'),            -1, -float('inf')],
            [           -7, -float('inf'),            -2],
            [-float('inf'), -float('inf'), -float('inf')],
            [           -5,            -3,            -6],
        ], dtype=tf.float32)

        active_mask = tf.constant([
            [True, False, True],
            [False, True, False],
            [True, True, True],
            [False, False, False],
        ], dtype=tf.bool)

        best_words = tf.constant([
            [  1, PAD,   9],
            [PAD,   2, PAD],
            [  3,   9,   4],
            [PAD, PAD, PAD],
        ], dtype=tf.int32)

        scores = tf.constant([
            [           10, -float('inf'),            14],
            [-float('inf'),            12, -float('inf')],
            [           11,            13,            15],
            [-float('inf'), -float('inf'), -float('inf')],
        ], dtype=tf.float32)

        tf.debugging.assert_equal(tf.shape(fin_storage_scores), [B, N])
        tf.debugging.assert_equal(tf.shape(active_mask),        [B, N])
        tf.debugging.assert_equal(tf.shape(best_words),         [B, N])
        tf.debugging.assert_equal(tf.shape(scores),             [B, N])


        exp_fin_storage_scores = tf.constant([
            [-float('inf'),            -1,            14],
            [           -7, -float('inf'),            -2],
            [-float('inf'),            13, -float('inf')],
            [           -5,            -3,            -6],
        ], dtype=tf.float32)

        exp_active_mask = tf.constant([
            [True, False, False],
            [False, True, False],
            [True, False, True],
            [False, False, False],
        ], dtype=tf.bool)

        tf.debugging.assert_equal(tf.shape(exp_fin_storage_scores), [B, N])
        tf.debugging.assert_equal(tf.shape(exp_active_mask),        [B, N])

        fin_storage_scores, active_mask = beam_search.update_fin(fin_storage_scores, best_words, active_mask, scores)

        exp_fin_storage_scores  = exp_fin_storage_scores.numpy()
        exp_active_mask         = exp_active_mask.numpy()

        np.testing.assert_array_equal(fin_storage_scores, exp_fin_storage_scores)
        np.testing.assert_array_equal(active_mask, exp_active_mask)
    
    def test_select_best_hyp(self):
        
        B = 3
        N = 4
        i = 5
        V = 10
        maxI = 8
        EOS = 9
        PAD = -1

        beam_search = self.__setup_test_object(B, maxI, N, V, EOS, PAD)

        tgt = tf.range(B*N*i, dtype=tf.int32)
        tgt = tf.reshape(tgt, (B, N, i))

        fin_storage_scores = tf.constant([
            [
                -4,
                -3,
                 0,
                -1
            ],
            [
                -float('inf'),
                -float('inf'),
                -float('inf'),
                -float('inf'),
            ],
            [
                -7,
                -5,
                -6,
                -5.1
            ],
        ], dtype=tf.float32)

        tf.debugging.assert_equal(tf.shape(tgt),                [B, N, i])
        tf.debugging.assert_equal(tf.shape(fin_storage_scores), [B, N])

        exp_tgt = tf.constant([
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
            [45, 46, 47, 48, 49],
        ], dtype=tf.int32)

        tgt = beam_search.select_best_hyp(tgt, fin_storage_scores)

        tgt      = tgt.numpy()
        exp_tgt  = exp_tgt.numpy()

        np.testing.assert_array_equal(tgt, exp_tgt)


    def __setup_test_object(self, B, maxI, N, V, EOS, PAD):

        hvd.init()

        path_test_vocab_src = 'tests/res/mockup_vocab_src.pickle'
        path_test_vocab_tgt = 'tests/res/mockup_vocab_tgt.pickle'

        config = Config(
            {
                'batch_size_search': B,
                'max_sentence_length': maxI,
                'beam_size': N,
                'length_norm': True,
                'model_dim': 20
            }
        )

        vocab_src = Vocabulary.create_vocab(path_test_vocab_src)
        vocab_tgt = Vocabulary.create_vocab(path_test_vocab_tgt)

        vocab_tgt.vocab_src = V
        vocab_tgt.vocab_size = V

        vocab_src.EOS = EOS
        vocab_tgt.EOS = EOS

        vocab_src.PAD = PAD
        vocab_tgt.PAD = PAD

        beam_search = BeamSearchFast.create_search_algorithm_from_config(config, Model(), vocab_src, vocab_tgt)

        return beam_search