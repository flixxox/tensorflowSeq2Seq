
from os import mkdir
from os.path import join, isdir

import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflowSeq2Seq.util.debug import my_print
from tensorflowSeq2Seq.util.globals import Globals


class Optimizer(tf.Module):

    def __init__(self, lr, numbers_dir, name='optimizer'):
        super(Optimizer, self).__init__(name=name)

        self.lr             = tf.Variable(lr, trainable=False)
        self.numbers_dir    = numbers_dir

    @staticmethod
    def create_optimizer_from_config(config, numbers_dir):

        if config['optimizer'] == 'WarmupAdam':
            
            return WarmupScheduledAdamOptimizer.create_optimizer_from_config(config, numbers_dir)

        else:

            assert True == False, 'Unknown score "%s"' % (config['score'])

    def write_lr_to_file(self, L):

        if hvd.rank() != 0:
            return

        if not isdir(self.numbers_dir):
            mkdir(self.numbers_dir)

        if isinstance(self.lr, tf.Variable):
            lr = float(self.lr.read_value().numpy())

        if isinstance(L, tf.Variable):
            L = float(L.read_value().numpy())

        with open(join(self.numbers_dir, 'lr'), 'a') as file:
            file.write(f'lr {lr}, L {L}\n')

    def step(self, grads, vars):
        raise NotImplementedError()

    def update_lr(self):
        raise NotImplementedError()


class WarmupScheduledAdamOptimizer(Optimizer):

    def __init__(self, numbers_dir, name='warmup_adam', **kwargs):
        super(WarmupScheduledAdamOptimizer, self).__init__(0., numbers_dir, name=name)

        with self.name_scope:

            for k, v in kwargs.items():
                setattr(self, k, v)

            self.adam = tf.keras.optimizers.Adam(
                learning_rate   = self.lr,
                beta_1          = 0.9,
                beta_2          = 0.98,
                epsilon         = 1e-9
            )

            self._step = tf.Variable(0, trainable=False)

    @staticmethod
    def create_optimizer_from_config(config, numbers_dir):
        return WarmupScheduledAdamOptimizer(
            numbers_dir,
            model_dim   = config["model_dim"], 
            warmup      = config["warmup"],
            update_freq = config["update_freq"],
            lr_scale    = config["lr_scale"],
        )

    @tf.function
    def step(self, grads, L, vars):

        def divide_sparse_dense(grads):

            sparse_grads = []
            dense_grads = []
            indicators = []

            for grad in grads:

                if isinstance(grad, tf.Tensor):
                    dense_grads.append(grad)
                    indicators.append('d')

                elif isinstance(grad, tf.IndexedSlices):
                    sparse_grads.append(grad)
                    indicators.append('s')

                else:
                    raise ValueError('Unrecognized gradient type.')

            return sparse_grads, dense_grads, indicators

        def merge_sparse_dense(sparse_grads, dense_grads, indicators):

            grads = []
            sparse_index = 0 
            dense_index = 0 

            for i, ind in enumerate(indicators):

                if ind == 's':
                    grads.append(sparse_grads[sparse_index])
                    sparse_index += 1

                elif ind == 'd':
                    grads.append(dense_grads[dense_index])
                    dense_index += 1
                
                else:
                    raise RuntimeError(f'Unrecognized gradient indicator. {ind}')

            assert sparse_index == len(sparse_grads)
            assert dense_index == len(dense_grads)

            return grads

        def average_grads(grads, L):

            avg_grads = []
        
            for grad in grads:

                if isinstance(grad, tf.Tensor):
                    avg_grads.append(grad / L)

                elif isinstance(grad, tf.IndexedSlices):
                    avg_grads.append(tf.IndexedSlices(indices=grad.indices, values=grad.values / L, dense_shape=grad.dense_shape))

                else:
                    raise ValueError('Unrecognized gradient type.')

            return avg_grads

        def update_lr():
            s   = tf.cast(self._step, tf.float32)
            w   = float(self.warmup)
            D   = float(self.model_dim)
            e1  = -0.5
            e2  = -1.5

            self.lr.assign((D ** e1) * tf.math.minimum(s ** e1, s * w ** e2) * self.lr_scale)

            self.adam.lr.assign(self.lr)

        self._step.assign_add(1)

        update_lr()

        sparse_grads, dense_grads, indicators = divide_sparse_dense(grads)

        dense_grads = hvd.grouped_allreduce(dense_grads, average=False)
        L = hvd.allreduce(L, average=False)

        for i, grad in enumerate(sparse_grads):
            sparse_grads[i] = tf.IndexedSlices(indices=hvd.allgather(grad.indices), values=hvd.allgather(grad.values), dense_shape=grad.dense_shape)

        grads = merge_sparse_dense(sparse_grads, dense_grads, indicators)

        average_grads(grads, L)

        self.adam.apply_gradients(zip(grads, vars))

        return self.lr