
from math import log
from os import mkdir
from os.path import join, isdir

import tensorflow as tf
import horovod.tensorflow as hvd


class Score(tf.Module):

    def __init__(self):
        super(Score, self).__init__()

    @staticmethod
    def create_score_from_config(config):

        if config['score'] == 'LabelSmoothingCrossEntropy':
            
            return LabelSmoothingCrossEntropyLoss.create_score_from_config(config)

        else:

            assert True == False, 'Unknown score "%s"' % (config['score'])

    @staticmethod
    def write_score_to_file(directory, filename, score):

        if hvd.rank() != 0:
            return

        if not isdir(directory):
            mkdir(directory)

        if isinstance(score, tf.Tensor):
            score = float(score.numpy())

        with open(join(directory, filename), 'a') as file:
            file.write(f'{score}\n')


class LabelSmoothingCrossEntropyLoss(Score):

    def __init__(self, m):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()

        self.m = m
        
        self.ce         = tf.Variable(0.)
        self.ce_smooth  = tf.Variable(0.)
        self.L          = tf.Variable(0.)

    @staticmethod
    def create_score_from_config(config):

        return LabelSmoothingCrossEntropyLoss(
            config['label_smoothing']
        )

    def __call__(self, output, out, out_mask=None):

        tgtV        = output.shape[-1]
        out_mask    = tf.cast(out_mask, dtype=tf.float32)

        out         = tf.reshape(out, (-1,1))
        output      = tf.reshape(output, (-1, tgtV))
        out_mask    = tf.reshape(out_mask, (-1,1))
        
        m = self.m
        w = m / (tgtV - 1)

        nll_loss = -1 * tf.gather(output, out, axis=-1, batch_dims=-1)
        smo_loss = -1 * tf.reduce_sum(output, axis=-1, keepdims=True)

        nll_loss = nll_loss * out_mask
        smo_loss = smo_loss * out_mask
        
        ce_smooth = (1 - m - w) * nll_loss + w * smo_loss

        num_words   = tf.reduce_sum(out_mask)
        ce_smooth   = tf.reduce_sum(ce_smooth)
        ce          = tf.reduce_sum(nll_loss)

        self.ce.assign_add(ce)
        self.ce_smooth.assign_add(ce_smooth)
        self.L.assign_add(num_words)

        return ce, ce_smooth, num_words

    def average_and_reset(self):

        ce          = hvd.allreduce(self.ce, average=False)
        ce_smooth   = hvd.allreduce(self.ce_smooth, average=False)
        L           = hvd.allreduce(self.L, average=False)

        ce          = float(ce.numpy())
        ce_smooth   = float(ce_smooth.numpy())
        L           = float(L.numpy())

        self.ce.assign(0.)
        self.ce_smooth.assign(0.)
        self.L.assign(0.)

        return (ce / L), (ce_smooth / L)


class BatchScore:
    """
    Calculates metrics that score the batching algorithm.
    
    1. Fullness: #non_padding_tokens/target_batch_size
    - Indicates how many tokens are used for training

    2. Padding Ratio: #padding_tokens/tokens_in_batch
    - Indicates how much memory is going to waste

    3. Randomness: Conditional entropy of batch_size_i+1 given batch_size_i
    - Indicates how random the batch algorithm samples the dataset
    - The higher the better
    """

    def __init__(self, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.fullness = tf.Variable(0., dtype=tf.float32)
        self.redundancy = tf.Variable(0., dtype=tf.float32)
        self.L = tf.Variable(0)

    @staticmethod
    def create_score_from_config(config, pad_index):

        return BatchScore(
            pad_index = pad_index,
            batch_size = config['batch_size'],
            maxI = config['max_sentence_length']
        )

    def __call__(self, tgt):
        
        fullness = self.__fullness(tgt)

        self.fullness.assign_add(fullness)

        redundancy = self.__redundancy(tgt)
        self.redundancy.assign_add(redundancy)

        self.L.assign_add(1)

    def average_and_reset(self):

        fullness    = hvd.allreduce(self.fullness, average=False)
        redundancy  = hvd.allreduce(self.redundancy, average=False)
        L           = hvd.allreduce(self.L, average=False)

        fullness    = float(fullness.numpy())
        redundancy  = float(redundancy.numpy())
        L           = float(L.numpy())

        self.fullness.assign(0.)
        self.redundancy.assign(0.)
        self.L.assign(0)

        return (fullness / L), (redundancy / L), 0.

    def __fullness(self, tensor):
        
        tensor = (tensor != self.pad_index)
        tensor = tf.cast(tensor, dtype=tf.float32)
        tensor = tf.reduce_sum(tensor)
        tensor =  tensor / self.batch_size

        return tensor

    def __redundancy(self, tensor):

        cur_batch_size = tensor.shape[0] * tensor.shape[1]

        tensor = (tensor == self.pad_index)
        tensor = tf.cast(tensor, dtype=tf.float32)
        tensor = tf.reduce_sum(tensor)
        tensor =  tensor / cur_batch_size

        return tensor
    
    def __randomness(self):

        cond_entropy = self.__calc_cond_entropy(self.I)

        return cond_entropy

    def __calc_cond_entropy(self, Is):
        
        zero    = [1e-9 for i in range(self.maxI)]
        joint   = [zero.copy() for i in range(self.maxI)]
        prior   = zero
        B       = len(Is)

        for i in range(B-1):
            c = Is[i]-1
            n = Is[i+1]-1

            joint[c][n] += 1.
            prior[c] += 1

        prior[Is[-1]-1] += 1

        joint = [[j / B for j in i] for i in joint]
        prior = [p / B for p in prior]

        entropy = 0

        for i in range(self.maxI):
            for j in range(self.maxI):
                entropy -= joint[i][j] * log((joint[i][j] / prior[i]), 2)

        return entropy