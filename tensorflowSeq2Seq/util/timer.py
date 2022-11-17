
import time

import tensorflow as tf

from tensorflowSeq2Seq.util.debug import my_print
from tensorflowSeq2Seq.util.globals import Globals

class Timer(tf.Module):
    def __init__(self, object_to_time):
        super(Timer, self).__init__(name=f'{object_to_time.name}_timer')

        with self.name_scope:

            self.object_to_time = object_to_time

            if Globals.do_timing():
                self.time_accum = tf.Variable(initial_value=0, dtype=tf.float64, trainable=False, name='time_accum')

    @staticmethod
    def print_timing_summary(model):

        timings = {}
        layer_summary = {}
        total_time = 0.
        encoder_time = 0.
        decoder_time = 0.

        for var in model.non_trainable_variables:
            if '_timer/time_accum' in var.name:
                value = float(var.read_value().numpy())
                total_time += value

                if var.name in timings:
                    timings[var.name] += value
                else:
                    timings[var.name] = value

        my_print('======== MEASUREMENTS ========')
        
        for k,v in timings.items():
            my_print(f'{k}'.ljust(130, ' '), f'{v:4.3f}s {(v/total_time)*100:4.1f}%')

        my_print('======== LAYER SUMMARY ========')

        for k,v in timings.items():
            if '_layer_' in k:
                k = k.split('/')
                k = '/'.join(k[:2] + k[3:])
                
                if k not in layer_summary:
                    layer_summary[k] = v
                else:
                    layer_summary[k] += v

        for k,v in layer_summary.items():
            my_print(f'{k}'.ljust(100, ' '), f'{v:4.3f}s {(v/total_time)*100:4.1f}%')

        my_print('======== ENCODER / DECODER SUMMARY ========')

        for k,v in timings.items():
            if '/encoder/' in k:
                encoder_time += v
            elif '/decoder/' in k:
                decoder_time += v
        
        my_print(f'Encoder'.ljust(20, ' '), f'{encoder_time:4.3f}s {(encoder_time/total_time)*100:4.1f}%')
        my_print(f'Decoder'.ljust(20, ' '), f'{decoder_time:4.3f}s {(decoder_time/total_time)*100:4.1f}%')

        my_print(f'Total Time: {total_time:4.3f}s')

        return total_time

    @staticmethod
    def timestamp():
        return time.perf_counter()

    def __call__(self, *args, **kwargs):

        if tf.autograph.experimental.do_not_convert(func=Globals.do_timing)():
            start = Timer.timestamp()
        
        out =  self.object_to_time(*args, **kwargs)

        if tf.autograph.experimental.do_not_convert(func=Globals.do_timing)():
            end = Timer.timestamp()
            self.time_accum.assign_add(end - start)

        return out

class ContextTimer:

    timings = {}

    def __init__(self, name):
        
        self.name = name
        self.timestamp_start = None

        if self.name not in ContextTimer.timings.keys():
            ContextTimer.timings[self.name] = 0.

    @staticmethod
    def print_summary(model_time=0):

        total_time = 0
        ContextTimer.timings['model'] = model_time

        my_print('======== ContextTimer MEASURMENTS ========')

        for k,v in ContextTimer.timings.items():
            total_time += v

        for k,v in ContextTimer.timings.items():
            my_print(f'{k}'.ljust(50, ' '), f'{v:4.3f}s {(v/total_time)*100:4.1f}%')

    def start(self):
        self.timestamp_start = Timer.timestamp()

    def end(self):
        ContextTimer.timings[self.name] += Timer.timestamp() - self.timestamp_start

    def __enter__(self):
        tf.autograph.experimental.do_not_convert(func=self.start)()

    def __exit__(self, exception_type, exception_value, traceback):
        tf.autograph.experimental.do_not_convert(func=self.end)()
