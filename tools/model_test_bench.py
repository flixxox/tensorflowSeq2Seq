import _setup_env

import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflowSeq2Seq.util.config import Config
from tensorflowSeq2Seq.model.models import Model
from tensorflowSeq2Seq.util.checkpoint_manager import CheckpointManager
from tensorflowSeq2Seq.util.data import (Vocabulary, Dataset)

# ======== CONFIG

config_file = 'EXAMPLE_CONFIG_FILE'

load_model = False
checkpoint_prefix = 'EXAMPLE_CHECKPOINT'

# ======== CREATION

hvd.init()

config = Config.parse_config({'config': config_file})

vocab_src = Vocabulary.create_vocab(config['vocab_src'])
vocab_tgt = Vocabulary.create_vocab(config['vocab_tgt'])

pad_index = vocab_src.PAD


model = Model.create_model_from_config(config,
    vocab_src.vocab_size,
    vocab_tgt.vocab_size,
)

epoch = tf.Variable(1)

if load_model:
    checkpoint_manager = CheckpointManager.create_eval_checkpoint_manager_from_config(config, epoch, model)
    checkpoint_manager.restore(checkpoint_prefix, partial=True)
else:
    model.init_weights(config['seed'])
    
# ======== CALLING

src = [
    ['</S>', 'lass', 'sie', 'kog@@', 'i', 'sein', '.', '&quot;', '</S>']
]

tgt = [
    ['</S>', 'let', '&apos;s', 'make']
]

src = [vocab_src.tokenize(s) for s in src]
tgt = [vocab_tgt.tokenize(t) for t in tgt]

print('tokenized src')
for s in src:
    print(s)

print('tokenized tgt')
for t in tgt:
    print(t)

src = tf.constant(src, dtype=tf.int32)
tgt = tf.constant(tgt, dtype=tf.int32)

masks, out_mask = model.create_masks(src, tgt, pad_index)

output, _ = model(src, tgt, **masks)

print(output.numpy())