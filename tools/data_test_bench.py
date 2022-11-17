import sre_compile
import _setup_env

import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflowSeq2Seq.util.config import Config
from tensorflowSeq2Seq.model.models import Model
from tensorflowSeq2Seq.util.setup import setup_tf_from_config
from tensorflowSeq2Seq.util.checkpoint_manager import CheckpointManager
from tensorflowSeq2Seq.util.data import Vocabulary, Dataset, BatchGenerator, BucketingBatchAlgorithm


# ======== CONFIG

config_file = 'EXAMPLE_CONFIG_FILE'

# ======== CREATION

hvd.init()

config = Config.parse_config(
    {
        'config': config_file,
        'force_eager_execution': False,
        'number_of_gpus': 2,
        'update_freq': 8,

    }
)

setup_tf_from_config(config)

vocab_src = Vocabulary.create_vocab(config['vocab_src'])
vocab_tgt = Vocabulary.create_vocab(config['vocab_tgt'])

dataset = Dataset.create_dataset_from_config(config, 'dataset', config['test_src'], config['test_tgt'], vocab_src, vocab_tgt)

batch_generator= BatchGenerator.create_batch_generator_from_config(config, dataset,  BucketingBatchAlgorithm)

pad_index = vocab_src.PAD
    
# ======== CALLING

for _, (src, tgt, out) in batch_generator.generate_batches():
    
    tgt = tgt.numpy()
    
    for s in tgt:
        s = list(s)
        s = vocab_src.detokenize_list(s)
        s = vocab_src.remove_padding(s)
        s = s[1:-1]
        s = ' '.join(s)

        print(s)