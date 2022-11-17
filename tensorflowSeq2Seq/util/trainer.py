import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflowSeq2Seq.util.globals import Globals
from tensorflowSeq2Seq.model.scores import BatchScore
from tensorflowSeq2Seq.util.timer import Timer, ContextTimer
from tensorflowSeq2Seq.util.debug import my_print, print_summary


class Trainer(tf.Module):

    def __init__(self, **kwargs):
        super(Trainer, self).__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.train_graph    = None
        self.eval_graph     = None

    @staticmethod
    def create_trainer_from_config(config, train_batch_generator, dev_batch_generator, model, criterion, optimizer):
        
        pad_index = train_batch_generator.dataset.vocab_src.PAD

        assert pad_index == train_batch_generator.dataset.vocab_tgt.PAD == dev_batch_generator.dataset.vocab_tgt.PAD == dev_batch_generator.dataset.vocab_src.PAD

        batch_score = None

        if config['score_batching']:
            batch_score = BatchScore.create_score_from_config(config, pad_index)

        trainer = Trainer(
            train_batch_generator   = train_batch_generator,
            dev_batch_generator     = dev_batch_generator,
            model                   = model,
            criterion               = criterion,
            optimizer               = optimizer,
            pad_index               = pad_index,
            batch_size              = config['batch_size'],
            update_freq             = config['update_freq'],
            seed                    = config['seed'],
            max_sentence_length     = config['max_sentence_length'],
            eager_execution         = (config['force_eager_execution'] or Globals.do_timing()),
            batch_score             = batch_score
        )

        trainer.trace()

        return trainer

    def trace(self):
        if self.eager_execution:
            self.__trace_eager()
        else:
            self.__trace_graph()
    
    def __trace_graph(self):

        my_print('Start tracing train!')

        self.train_graph = self.train_step.get_concrete_function(
            src=tuple(tf.TensorSpec((None, None), tf.int32) for _ in range(self.update_freq)),
            tgt=tuple(tf.TensorSpec((None, None), tf.int32) for _ in range(self.update_freq)),
            out=tuple(tf.TensorSpec((None, None), tf.int32) for _ in range(self.update_freq))
        )

        my_print('Start tracing eval!')

        self.eval_graph = self.eval_get_scores.get_concrete_function(
            src=tf.TensorSpec((None, None), tf.int32),
            tgt=tf.TensorSpec((None, None), tf.int32),
            out=tf.TensorSpec((None, None), tf.int32)
        )

    def __trace_eager(self):

        self.train_graph    = self.train_step
        self.eval_graph     = self.eval_get_scores
        
    def train(self, epoch):

        assert self.train_graph is not None

        step = 1
        max_steps = None

        for _, (src, tgt, out), total_steps in self.train_batch_generator.generate_batches():

            assert len(src) == len(tgt) == len(out) == self.update_freq

            if max_steps is None:
                max_steps = min(hvd.allgather_object(total_steps, name=f'gather_steps_train'))
            elif step > max_steps:
                break

            (lr, L) = self.train_graph(src, tgt, out)

            self.optimizer.write_lr_to_file(L)

            if self.batch_score is not None:
                self.batch_score(tgt)

            step += 1

        if Globals.do_timing():
            model_time = Timer.print_timing_summary(self.model)
            ContextTimer.print_summary(model_time)

        ce, ce_smooth   = self.criterion.average_and_reset()
        ppl, ppl_smooth = self.__calculate_ppl(ce, ce_smooth)

        to_print = {
            'ce':               ce,
            'ce_smooth':        ce_smooth,
            'ppl':              ppl,
            'ppl_smooth':       ppl_smooth,
            'b_fullness':       None,
            'b_padding_ratio':  None,
            'steps':            step,
            'lr':               lr,
        }

        if self.batch_score is not None:
            to_print['b_fullness'], to_print['b_padding_ratio'], _ = self.batch_score.average_and_reset()

        print_summary(True, epoch, **to_print)

        return ppl
    
    def eval(self, epoch):

        assert self.eval_graph is not None

        step        = 0
        max_steps   = None

        for _, (src, tgt, out), total_steps in self.dev_batch_generator.generate_batches():

            if max_steps is None:
                max_steps = min(hvd.allgather_object(total_steps, name=f'gather_steps_eval'))
            elif step > max_steps:
                break

            ce, ce_smooth, _ = self.eval_graph(src, tgt, out)

            if self.batch_score is not None:
                self.batch_score(tgt)

            step += 1

        ce, ce_smooth       = self.criterion.average_and_reset()
        ppl, ppl_smooth     = self.__calculate_ppl(ce, ce_smooth)

        to_print = {
            'ce':               ce,
            'ce_smooth':        ce_smooth,
            'ppl':              ppl,
            'ppl_smooth':       ppl_smooth,
            'b_fullness':       None,
            'b_padding_ratio':  None,
            'steps':            step
        }

        if self.batch_score is not None:
            to_print['b_fullness'], to_print['b_padding_ratio'], _ = self.batch_score.average_and_reset()

        print_summary(False, epoch, **to_print)

        return ppl

    def __calculate_ppl(self, ce, ce_smooth):

        try: 
            ppl         = tf.math.exp(ce) 
            ppl_smooth  = tf.math.exp(ce_smooth)

            ppl         = float(ppl.numpy())
            ppl_smooth  = float(ppl_smooth.numpy())

        except OverflowError: 
            ppl         = float('inf')
            ppl_smooth  = float('inf')

        return ppl, ppl_smooth

    @tf.function
    def train_step(self, src, tgt, out):

        def accumulate_grads(grads_accum, grads):

            for i in range(len(grads)):

                if isinstance(grads[i], tf.Tensor):
                    grads_accum[i] += grads[i]

                elif isinstance(grads[i], tf.IndexedSlices):
                    values = tf.concat([grads_accum[i].values, grads[i].values], axis=0)
                    indices = tf.concat([grads_accum[i].indices, grads[i].indices], axis=0)
                    grads_accum[i] = tf.IndexedSlices(values, indices, dense_shape=grads_accum[i].dense_shape)

                else:
                    raise ValueError(f'Unrecognized gradient type. {type(grads[i])}')

        vars = self.model.trainable_variables

        L_accum, grads_accum = self.train_ministep(src[0], tgt[0], out[0])
        grads_accum = list(grads_accum)

        for i in range(1, len(src)):
            L, grads = self.train_ministep(src[i], tgt[i], out[i])

            accumulate_grads(grads_accum, grads)
            L_accum += L

        with ContextTimer('optimizer_step'):
            lr = self.optimizer.step(grads_accum, L_accum, vars)

        return (lr, L_accum)
    
    @tf.function
    def train_ministep(self, src, tgt, out):

        with ContextTimer('model_mask_creation'):
            masks, out_mask = self.model.create_masks(src, out, self.pad_index)

        with tf.GradientTape() as tape:

            output, _ = self.model(src, tgt, training=True, **masks)

            with ContextTimer('criterion'):
                _, ce_smooth, L_ce = self.criterion(output, out, out_mask=out_mask)

        with ContextTimer('backpropagation'):
            grads = tape.gradient(ce_smooth, self.model.trainable_variables)

        return L_ce, grads

    @tf.function
    def eval_get_scores(self, src, tgt, out):

        masks, out_mask     = self.model.create_masks(src, out, self.pad_index)
        output, _           = self.model(src, tgt, training=False, **masks)
        ce, ce_smooth, L_ce = self.criterion(output, out, out_mask=out_mask)

        return ce, ce_smooth, L_ce

