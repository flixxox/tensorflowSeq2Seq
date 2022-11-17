
import tensorflow as tf

from tensorflowSeq2Seq.util.debug import my_print
from tensorflowSeq2Seq.search.search_algorithm import SearchAlgorithm


class BeamSearchLong(SearchAlgorithm):

    def __init__(self, name='beam_search', **kwargs):
        super(BeamSearchLong, self).__init__(name=name, **kwargs)

    @staticmethod
    def create_search_algorithm_from_config(config, model, vocab_src, vocab_tgt):

        return BeamSearchLong(
            model       = model,
            vocab_src   = vocab_src,
            vocab_tgt   = vocab_tgt,
            D           = config['model_dim'],
            maxI        = config['max_sentence_length'],
            beam_size   = config['beam_size'],
            length_norm = config['length_norm']
        )

    @tf.function
    def search_batch(self, src):

        B       = tf.shape(src)[0]
        srcT    = tf.shape(src)[1]
        maxI    = tf.cast(tf.math.minimum(tf.math.round(tf.cast(srcT, dtype=tf.float32)*1.25), self.maxI), dtype=tf.int32)
        N       = self.beam_size
        D       = self.D
        BN      = B * N
        V       = self.V
        model   = self.model
        PAD     = self.PAD
        EOS     = self.EOS

        filler_tensor       = tf.ones([B, N, maxI], dtype=tf.int32) * PAD
        fin_storage         = tf.ones([B, N, maxI], dtype=tf.int32) * PAD
        fin_storage_scores  = tf.ones([B, N]) * -float("inf")
        tgt                 = tf.ones((B, 1), dtype=tf.int32) * EOS
        
        tf.debugging.assert_equal(tf.shape(tgt), [B, 1])
        tf.debugging.assert_equal(tf.shape(src), [B, srcT])

        masks, _    = self.model.create_masks(src, tgt, PAD)
        h           = self.model.encoder(src, **masks)
        output      = self.model.decoder(tgt, h, **masks)

        tf.debugging.assert_equal(tf.shape(output), [B, 1, V])

        _, output, scores, fin_scores = self._select_n_best(output)

        tf.debugging.assert_equal(tf.shape(output),     [B, N])
        tf.debugging.assert_equal(tf.shape(scores),     [B, N])
        tf.debugging.assert_equal(tf.shape(fin_scores), [B, 1])

        tgt = tf.repeat(tgt, N, axis=1)
        tgt = tf.expand_dims(tgt, axis=-1)

        fin_storage, fin_storage_scores = self._append_finished(tgt, fin_storage, fin_storage_scores, fin_scores, filler_tensor)

        tf.debugging.assert_equal(tf.shape(fin_storage),        [B, N, maxI])
        tf.debugging.assert_equal(tf.shape(fin_storage_scores), [B, N])
        tf.debugging.assert_equal(tf.shape(tgt),                [B, N, 1])

        output = tf.expand_dims(output, axis=-1)

        tgt = tf.concat((tgt, output), axis=-1)
        tgt = tf.reshape(tgt, (BN, 2))

        src = tf.expand_dims(src, axis=1)
        src = tf.repeat(src, N, axis=1)
        src = tf.reshape(src, (BN, srcT))

        h = tf.expand_dims(h, axis=1)
        h = tf.repeat(h, N, axis=1)
        h = tf.reshape(h, (BN, srcT, D))

        tf.debugging.assert_equal(tf.shape(tgt),    [BN, 2])
        tf.debugging.assert_equal(tf.shape(scores), [B, N])
        tf.debugging.assert_equal(tf.shape(src),    [BN, srcT])
        tf.debugging.assert_equal(tf.shape(h),      [BN, srcT, D])

        i = tf.constant(2, dtype=tf.int32)
        while i <= maxI:
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (tgt,   tf.TensorShape([None, None]))
                ]
            )

            tf.debugging.assert_equal(tf.shape(tgt), [BN, i])

            masks, _    = self.model.create_masks(src, tgt, PAD)
            output      = self.model.decoder(tgt, h, **masks) # [BNa, i, V]

            output = output[:,-1,:]

            tf.debugging.assert_equal(tf.shape(output), [BN, V])

            output = tf.reshape(output, (B, N, V))
            scores = tf.reshape(scores, (B, N, 1))

            output = output + scores

            if self.length_norm:
                output /= tf.cast(i, dtype=tf.float32)

            best_beams, best_words, scores, fin_scores = self._select_n_best(output)

            if self.length_norm:
                scores *= tf.cast(i, dtype=tf.float32)

            tf.debugging.assert_equal(tf.shape(best_beams), [B, N])
            tf.debugging.assert_equal(tf.shape(best_words), [B, N])
            tf.debugging.assert_equal(tf.shape(scores),     [B, N])
            tf.debugging.assert_equal(tf.shape(fin_scores), [B, N])

            tgt = tf.reshape(tgt, (B, N, i))

            fin_storage, fin_storage_scores = self._append_finished(tgt, fin_storage, fin_storage_scores, fin_scores, filler_tensor)
            tgt                             = self._append_tgt(tgt, best_beams, best_words)

            i += 1

            tgt = tf.reshape(tgt, (BN, i))

            tf.debugging.assert_equal(tf.shape(tgt),                [BN, i])
            tf.debugging.assert_equal(tf.shape(fin_storage),        [B, N, maxI])
            tf.debugging.assert_equal(tf.shape(fin_storage_scores), [B, N])

        return fin_storage[:,0,:]


    def _select_n_best(self, output):

        B       = tf.shape(output)[0]
        V       = tf.shape(output)[-1]
        N_in    = tf.shape(output)[1]
        N_out   = self.beam_size

        fin_scores  = tf.gather(output, self.EOS, axis=-1)
        fin_scores  = tf.reshape(fin_scores, (B, N_in))

        output = tf.where(self.fin_V_mask, -float('inf'), output)
        output = tf.reshape(output, (B, N_in*V))

        best_scores, best_indices = tf.math.top_k(output, k=N_out)
        best_words = best_indices % V
        best_beams = best_indices // V

        smallest_best = tf.reduce_min(output, axis=-1, keepdims=True)
        smallest_best = tf.repeat(smallest_best, N_in, axis=-1)

        fin_score_mask  = (fin_scores < smallest_best)
        fin_scores      = tf.where(fin_score_mask, -float('inf'), fin_scores)

        return best_beams, best_words, best_scores, fin_scores

    def _append_finished(self, tgt, fin_storage, fin_storage_scores, fin_scores, filler_tensor):
        
        B       = tf.shape(tgt)[0]
        i       = tf.shape(tgt)[-1]
        maxI    = tf.shape(filler_tensor)[-1]
        N       = self.beam_size

        fin_storage_scores              = tf.concat((fin_scores, fin_storage_scores), axis=-1)
        fin_storage_scores, fin_indices = tf.math.top_k(fin_storage_scores, k=N)

        tf.debugging.assert_equal(tf.shape(fin_storage_scores), [B, N])

        tgt_tmp = tf.concat((tgt, filler_tensor[:,:,:maxI-i]), axis=-1)

        tf.debugging.assert_equal(tf.shape(tgt_tmp), tf.shape(fin_storage))

        fin_storage = tf.concat((tgt_tmp, fin_storage), axis=1)
        fin_storage = tf.gather(fin_storage, fin_indices, axis=1, batch_dims=1)

        return fin_storage, fin_storage_scores

    def _append_tgt(self, tgt, best_beams, best_words):

        B = tf.shape(tgt)[0]
        N = self.beam_size

        tgt = tf.gather(tgt, best_beams, axis=1, batch_dims=1)

        best_words  = tf.reshape(best_words, (B, N, 1))
        tgt         = tf.concat((tgt, best_words), axis=-1)

        return tgt