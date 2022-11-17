
import tensorflow as tf

from tensorflowSeq2Seq.util.debug import my_print
from tensorflowSeq2Seq.search.search_algorithm import SearchAlgorithm


class BeamSearchFast(SearchAlgorithm):

    def __init__(self, name='beam_search', **kwargs):
        super(BeamSearchFast, self).__init__(name=name, **kwargs)

    @staticmethod
    def create_search_algorithm_from_config(config, model, vocab_src, vocab_tgt):

        return BeamSearchFast(
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
        """
        TODO: tgt to TensorArray
        """

        B       = tf.shape(src)[0]
        srcT    = tf.shape(src)[1]
        N       = self.beam_size
        D       = self.D
        BN      = B*N     
        maxI    = self.maxI
        V       = self.V
        PAD     = self.PAD
        EOS     = self.EOS

        fin_storage_scores  = tf.ones((B, N), dtype=tf.float32)
        active_mask         = tf.ones((B, N), dtype=tf.bool)
        BNa                 = BN
        tgt                 = tf.ones((B, N, 1), dtype=tf.int32) * EOS

        precomp_indices = tf.expand_dims(tf.range(N), axis=0)
        precomp_indices = tf.repeat(precomp_indices, B, axis=0) + tf.expand_dims(tf.range(B), axis=1) * N

        tgt_first   = tf.ones((B, 1), dtype=tf.int32) * EOS
        
        masks, _    = self.model.create_masks(src, tgt_first, PAD)
        encs           = self.model.encoder(src, **masks)

        if not isinstance(encs, (list, tuple)):
            encs = [encs]
        elif isinstance(encs, tuple):
            encs = list(encs)

        output      = self.model.decoder(tgt_first, *encs, **masks)

        output              = tf.squeeze(output, axis=1)
        scores, best_words  = tf.math.top_k(output, k=N)
        best_words          = tf.expand_dims(best_words, axis=-1)
        tgt                 = tf.concat((tgt, best_words), axis=-1)

        src = tf.expand_dims(src, axis=1)
        src = tf.repeat(src, N, axis=1)

        for i, _ in enumerate(encs):
            encs[i] = tf.expand_dims(encs[i], axis=1)
            encs[i] = tf.repeat(encs[i], N, axis=1)
            tf.debugging.assert_equal(tf.shape(encs[i]), [B, N, srcT, D])

        tf.debugging.assert_equal(tf.shape(precomp_indices),    [B, N])
        tf.debugging.assert_equal(tf.shape(scores),             [B, N])
        tf.debugging.assert_equal(tf.shape(src),                [B, N, srcT])
        tf.debugging.assert_equal(tf.shape(tgt),                [B, N, 2])

        i = tf.constant(2, dtype=tf.int32)
        while BNa > 0 and i <= maxI:
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (tgt,   tf.TensorShape([None, N, None]))
                ]
            )

            srca, encsa, tgta = self.get_active_model_input(src, encs, tgt, active_mask)

            tf.debugging.assert_equal(tf.shape(srca),   [BNa, srcT])
            tf.debugging.assert_equal(tf.shape(tgta),   [BNa, i])

            masks, _    = self.model.create_masks(srca, tgta, PAD)
            output      = self.model.decoder(tgta, *encsa, **masks) # [BNa, i, V]

            tf.debugging.assert_equal(tf.shape(output), [BNa, i, V])

            output = output[:,-1,:] # [BNa, V]

            tf.debugging.assert_equal(tf.shape(output), [BNa, V])

            output = self.pad_to_N(output, active_mask, precomp_indices)

            tf.debugging.assert_equal(tf.shape(output), [B, N, V])

            output += tf.expand_dims(scores, axis=-1)
            output = self.apply_length_norm(output, i)
            output = tf.reshape(output, (B, N*V))

            scores, best_words  = tf.math.top_k(output, k=N)
            best_beams          = best_words // V
            best_words          = best_words % V

            tf.debugging.assert_equal(tf.shape(scores),     [B, N])
            tf.debugging.assert_equal(tf.shape(best_beams), [B, N])
            tf.debugging.assert_equal(tf.shape(best_words), [B, N])

            tgt, scores, best_words         = self.update_tgt(tgt, active_mask, best_beams, best_words, scores)

            tf.debugging.assert_equal(tf.shape(tgt),        [B, N, i+1])
            tf.debugging.assert_equal(tf.shape(scores),     [B, N])
            tf.debugging.assert_equal(tf.shape(best_words), [B, N])

            fin_storage_scores, active_mask = self.update_fin(fin_storage_scores, best_words, active_mask, scores)

            tf.debugging.assert_equal(tf.shape(fin_storage_scores), [B, N])
            tf.debugging.assert_equal(tf.shape(active_mask),        [B, N])

            BNa                             = self.update_BNa(active_mask)

            scores = self.remove_length_norm(scores, i)

            i += 1

        tgt = self.select_best_hyp(tgt, fin_storage_scores)

        tf.debugging.assert_equal(tf.shape(tgt), [B, i])

        return tgt

    def get_active_model_input(self, src, encs, tgt, active_mask):

        srca = tf.boolean_mask(src, active_mask)
        encsa = [tf.boolean_mask(enc, active_mask) for enc in encs]
        tgta = tf.boolean_mask(tgt, active_mask)

        return srca, encsa, tgta

    def pad_to_N(self, output, active_mask, precomp_indices):
        
        B = tf.shape(active_mask)[0]
        N = tf.shape(active_mask)[1]
        V = tf.shape(output)[-1]

        indices = tf.boolean_mask(precomp_indices, active_mask)
        indices = tf.expand_dims(indices, axis=-1)

        output = tf.scatter_nd(indices, output, [B*N, V])
        output = tf.reshape(output, (B, N, V))
        output = tf.where(tf.expand_dims(tf.math.logical_not(active_mask), axis=-1), -float('inf'), output)

        return output

    def apply_length_norm(self, output, i):
        if self.length_norm:
            return output / tf.cast(i, dtype=tf.float32)
        else:
            return output

    def update_tgt(self, tgt, active_mask, best_beams, best_words, scores):
    
        indices         = tf.cumsum(tf.cast(active_mask, dtype=tf.int32), axis=-1, exclusive=True)
        rev_active_mask = active_mask == False

        best_words = tf.gather(best_words, indices, axis=-1, batch_dims=1)
        best_words = tf.where(rev_active_mask, self.PAD, best_words)

        best_beams = tf.gather(best_beams, indices, axis=-1, batch_dims=1)
        best_beams = tf.where(rev_active_mask, tf.range(self.beam_size), best_beams)

        tgt = tf.gather(tgt, best_beams, axis=1, batch_dims=1)
        tgt = tf.concat((tgt, tf.expand_dims(best_words, axis=-1)), axis=-1)

        scores = tf.gather(scores, indices, axis=-1, batch_dims=1)
        scores = tf.where(rev_active_mask, -float('inf'), scores)

        return tgt, scores, best_words

    def update_fin(self, fin_storage_scores, best_words, active_mask, scores):

        cur_fin_mask            = best_words == self.EOS
        fin_storage_scores      = tf.where(cur_fin_mask, scores, fin_storage_scores)

        active_mask = tf.math.logical_and(cur_fin_mask == False, active_mask)

        return fin_storage_scores, active_mask
    
    def update_BNa(self, active_mask):
        return tf.reduce_sum(tf.cast(active_mask, dtype=tf.int32))

    def remove_length_norm(self, output, i):
        if self.length_norm:
            return output * tf.cast(i, dtype=tf.float32)
        else:
            return output

    def select_best_hyp(self, tgt, fin_storage_scores):

        _, indices = tf.math.top_k(fin_storage_scores, k=1)

        tgt = tf.gather(tgt, indices, batch_dims=1)
        tgt = tf.squeeze(tgt, axis=1)

        return tgt