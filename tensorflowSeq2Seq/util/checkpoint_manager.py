from os import mkdir, listdir
from os.path import isdir, isfile, join

import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflowSeq2Seq.util.debug import my_print

class CheckpointManager:

    def __init__(self, managed_items, **kwargs):
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.best_ppl           = tf.Variable(float('inf'))
        self.ckpts_since_best   = tf.Variable(0)
        self.checkpoint         = tf.train.Checkpoint(best_ppl=self.best_ppl, **managed_items)

    @staticmethod
    def create_train_checkpoint_manager_from_config(config, epoch, optimizer, model):

        if config['resume_training']:
            config['output_folder'] = join(config['resume_training_from'], 'output')

        checkpoint_dir = join(config['output_folder'], 'checkpoints')

        if not isdir(checkpoint_dir):
            mkdir(checkpoint_dir)

        managed_items = {
            'epoch': epoch,
            'optimizer': optimizer,
            'model': model
        }

        checkpoint_manager = CheckpointManager(
            managed_items,
            checkpoint_dir                  = checkpoint_dir,
            save_freq                       = config['checkpoint_frequency'],
            resume_training                 = config['resume_training'],
            do_checkpoints                  = config['checkpoints'],
            checkpoint_strategy             = config['checkpoint_strategy'],
            do_early_abort                  = config['early_abort'],
            ckpts_till_abort                = config['ckpts_till_abort'],
            delay_checkpointing_to_epoch    = config['delay_checkpointing_to_epoch', 0],
            load_weights                    = config['load_weights', False],
            load_weights_from               = config['load_weights_from', ""]
        )

        return checkpoint_manager

    @staticmethod
    def create_eval_checkpoint_manager_from_config(config, epoch, model):

        managed_items = {
            'epoch': epoch,
            'model': model
        }

        checkpoint_manager = CheckpointManager(
            managed_items
        )

        return checkpoint_manager

    def restore_or_initialize(self, model, seed=None):
        if self.resume_training:
            self.restore_latest()
        elif self.load_weights:
            self.load_weights_from_checkpoint(model)
        else:
            model.init_weights(seed)

    def restore_latest(self):
        self.restore(self.get_latest_checkpoint_path())
    
    def load_weights_from_checkpoint(self, model):

        my_print(f'Loading weights from {self.load_weights_from}')

        model_checkpoint = tf.train.Checkpoint(model=model)
        model_checkpoint.read(self.load_weights_from).expect_partial()

    def get_latest_checkpoint_path(self):
        
        file_names = [f for f in listdir(self.checkpoint_dir) if isfile(join(self.checkpoint_dir, f))]
        max_number = -1

        for file_name in file_names:
            
            if file_name.startswith('ckpt-'):
                number = file_name.split('.')[0].split('-')[1]

                if number.isdigit():
                    number = int(number)

                    if number > max_number:
                        max_number = number

        return join(self.checkpoint_dir, f'ckpt-{max_number}')

    def restore(self, path, partial=False):

        my_print(f'Loading weights from {path}')

        if partial:
            self.checkpoint.read(path).expect_partial()
        else:
            self.checkpoint.read(path)

    def save(self, number, ppl):

        if not self.do_checkpoints:
            return

        if hvd.rank() != 0:
            return

        if isinstance(number, tf.Variable):
            number = number.read_value()
        
        if isinstance(number, tf.Tensor):
            number = int(number.numpy())

        if number < self.delay_checkpointing_to_epoch:
            return

        if self.checkpoint_strategy == 'All':
            if number % self.save_freq == 0:
                self.save_epoch(number)
        
        if ppl < self.best_ppl:
            self.save_best()
            self.best_ppl = ppl
            self.ckpts_since_best.assign(0)
        else:
            self.ckpts_since_best.assign_add(1)

    def save_epoch(self, number):
        my_print('Saving epoch checkpoint!')
        self.__save(join(self.checkpoint_dir, f'ckpt-{number}'))

    def save_best(self):
        my_print('Saving best checkpoint!')
        self.__save(join(self.checkpoint_dir, f'ckpt-best'))

    def __save(self, path):
        self.checkpoint.write(path)

    def early_abort(self):
        
        if not self.do_early_abort:
            return False

        if self.ckpts_since_best >= self.ckpts_till_abort:
            return True
        else:
            return False
        


