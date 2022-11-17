
def my_print(*args, **kwargs):

    import horovod.tensorflow as hvd

    if hvd.rank() == 0:
        print(*args, flush=True, **kwargs)

def tf_num_gpus_available():
    import tensorflow as tf
    return len(tf.config.list_physical_devices('GPU'))

def get_number_of_trainable_variables(model):
    sum = 0
    for var in model.trainable_variables:
        cur = 1
        for s in var.shape:
            cur *= s
        sum += cur
    return sum

def tf_print_memory_usage():
    import tensorflow as tf
    import horovod.tensorflow as hvd
    from tensorflowSeq2Seq.util.globals import Globals

    if Globals.is_gpu():
        mem_info = tf.config.experimental.get_memory_info('GPU:0')
        print(f'Worker {hvd.rank()}: Current memory usage: {(mem_info["current"] / 1e9):4.2f}GB, Peak memory usage: {(mem_info["peak"] / 1e9):4.2f}GB', flush=True)
    else:
        my_print('Showing memory usage only available on GPU!')

def print_summary(is_train, epoch, **kwargs):

    import tensorflow as tf
    
    def pr_int(name, value):
        length = len(name) + 2 + 4
        if value is not None:
            return f'{name}: {value:0>4}'
        else:
            return ''.center(length, ' ')

    def pr_float_precise(name, value):
        length = len(name) + 2 + 10 + 6
        if value is not None:
            if value > 1e8:
                value = float('inf')
            return f'{name}: {value:8.6f}'.ljust(length, ' ')
        else:
            return ''.center(length, ' ')

    def pr_float(name, value):
        length = len(name) + 2 + 8 + 2
        if value is not None:
            if value > 1e7:
                value = float('inf')
            return f'{name}: {value:4.2f}'.ljust(length, ' ')
        else:
            return ''.center(length, ' ')

    first_choices = ['train', 'eval']

    if is_train:
        first = first_choices[0]
    else:
        first = first_choices[1]
    first_length = max([len(s) for s in first_choices])

    to_print = (
        f'| {first.center(first_length, " ")} '
        f'| epoch: {epoch:0>4} '
    )

    for k, v in kwargs.items():

        if v is None:
            continue

        if isinstance(v, tf.Variable):
            v = v.read_value()

        if isinstance(v, tf.Tensor):
            v = v.numpy()
            if v == int(v):
                v = int(v)
            else:
                v = float(v)

        if isinstance(v, int):
            to_print = (
                f'{to_print}'
                f'| {pr_int(k, v)} '
            )
        elif isinstance(v, float):
            if v > 1e-2 or v == 0.:
                to_print = (
                    f'{to_print}'
                    f'| {pr_float(k, v)} '
                )
            else:
                to_print = (
                    f'{to_print}'
                    f'| {pr_float_precise(k, v)} '
                )

    my_print(to_print)