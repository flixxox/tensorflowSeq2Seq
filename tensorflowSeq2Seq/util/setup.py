
import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflowSeq2Seq.util.globals import Globals
from tensorflowSeq2Seq.util.debug import my_print, tf_num_gpus_available, tf_print_memory_usage

def setup_tf_from_config(config):

    if config['force_eager_execution'] or Globals.do_timing():
        my_print('Eager execution enabled!')
        tf.config.run_functions_eagerly(True)
        Globals.set_eager_flag(True)

        if Globals.do_timing():
            tf.config.experimental.set_synchronous_execution(True)        

    my_print('Available devices:', tf.config.list_physical_devices())

    num_gpus_avail = tf_num_gpus_available()

    my_print(f'Number of GPUs available: {num_gpus_avail}')

    config['number_of_gpus'] = max(0, config['number_of_gpus'])

    assert config['number_of_gpus'] <= num_gpus_avail, f'Not enough GPUs available! Avail: {num_gpus_avail}, Requested {config["number_of_gpus"]}'

    Globals.set_number_of_workers(max(1, config['number_of_gpus']))

    if config['number_of_gpus'] <= 0:
        my_print('Limiting to CPU!')
        tf.config.set_visible_devices([], 'GPU')
        Globals.set_cpu()

    else:

        setup_horovod(config)
        tf_print_memory_usage()

def setup_horovod(config):

    my_print('Setting up Horovod!')

    config['update_freq'] = config['update_freq'] // Globals.get_number_of_workers()

    my_print(f'Scaled down update_freq to {config["update_freq"]}!')

    gpus = tf.config.experimental.list_physical_devices('GPU')[:config['number_of_gpus']]

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    my_print(f'Horovod: Is MPI enabled at runtime? {hvd.mpi_enabled()}! Hvd build with MPI? {hvd.mpi_built()}!')
    my_print(f'Horovod: Hvd compiled with nccl support? {hvd.nccl_built()}! Hvd build with cuda support? {hvd.cuda_built()}!')