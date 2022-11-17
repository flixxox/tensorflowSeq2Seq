
import tensorflow as tf

from tensorflowSeq2Seq.search.search_algorithm import SearchAlgorithm


class GreedySearch(SearchAlgorithm):

    def __init__(self, name='greedy', **kwargs):
        super(GreedySearch, self).__init__(name=name, **kwargs)

    @staticmethod
    def create_search_algorithm_from_config(config, model, vocab_src, vocab_tgt):

        return GreedySearch(
            model = model,
            vocab_src = vocab_src,
            vocab_tgt = vocab_tgt,
            batch_size = config['batch_size_search'],
            maxI = config['max_sentence_length']
        )

    @tf.function
    def search_batch(self, src):

        B = tf.shape(src)[0]
        maxI = self.maxI
        V = self.vocab_tgt.vocab_size
        finished_batches = tf.zeros(B, dtype=tf.bool)
        i = tf.constant(1, dtype=tf.int32)

        eos_index_tgt = self.vocab_tgt.EOS
        pad_index = self.vocab_src.PAD

        end_tensor = tf.constant([eos_index_tgt], dtype=tf.int32)
        end_tensor = tf.repeat(end_tensor, repeats=B, axis=0) # [B]

        tgt = tf.constant([eos_index_tgt] + [pad_index for j in range(maxI-1)], dtype=tf.int32) # [maxI]
        tgt = tf.expand_dims(tgt, axis=0) # [1, maxI]
        tgt = tf.repeat(tgt, repeats=B, axis=0) # [B, maxI]

        while not tf.reduce_all(finished_batches) and i <= self.maxI:

            # tgt.shape == [B, maxI]

            cur_tgt = tgt[:,:i] # TFLite does not like this. Requires Select OPS

            masks, _ = self.model.create_masks(src, cur_tgt, pad_index)
            output, _ = self.model(src, cur_tgt, **masks) # [B, i, V]

            output = output[:,-1,:]
            output = tf.reshape(output, [B, V]) # [B, V]

            output, indices = tf.math.top_k(output, k=1) # [B, 1]

            indices = tf.reshape(indices, [B]) # [B]

            finished_mask = (indices == end_tensor) # [B, 1]
            finished_mask = tf.reshape(finished_mask, [B]) # [B]

            finished_batches = tf.math.logical_or(finished_batches, finished_mask)

            tf.debugging.assert_shapes([(finished_batches, ('B')), (indices, ('B'))])

            indices = tf.where(finished_batches, pad_index, indices) # [B]

            indices = tf.expand_dims(indices, axis=1) # [B, 1]

            tgt = tf.where(
                (tf.range(maxI) == i),
                indices,
                tgt
            )

            i += 1

        return tgt