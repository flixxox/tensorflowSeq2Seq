# tensorflowSeq2Seq
This repository is used to train and search NMT models created with Tensorflow.
It allows to easily define the model, and if needed optimizers and scores, using native Tensorflow API.
Furthermore, it takes care of other functionalities used in training and search in an easy to understand manner.

## Features

- Easy to define models, optimizers and scores
- The code is lightweight and easy to understand
- Multi-threaded data loading
- Data-parallel multi-gpu training using Horovod
- Implementation for Transformer (Base + Big)
- Implementation for BeamSearch (Fast + Long) and GreedySearch
- Checkpoint manager implementing checkpoint delay, checkpoint frequency and checkpoint strategies ALL and BEST
- TFLite compatibility for most models and all search algorithms
- Provides an interface for time measurements of modules


### Train

- Save checkpoints within a checkpoint frequency
- Saving the best checkpoint according to any score
- Using graph mode for major operations: calculate gradients, call model, search model. The train loop as a whole is not applied in graph mode
- Eager execution is also provided but not recommended

### Models

- Transformer (Base + Big) ([Attention Is All You Need](https://arxiv.org/abs/1706.03762))
- Transforer with relative positional representations ([Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155))
- FNet ([FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824)) adopted to MT
- MLP-Mixer ([MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)) adopted to MT
- Multi-head NN ([Are Neighbors Enough? Multi-Head Neural n-gram can be Alternative to Self-attention](https://arxiv.org/abs/2207.13354)) 

### Optimizers

- We use the TF base implementation of Adam with a warmup learning rate scheduler according to ([Attention Is All You Need](https://arxiv.org/abs/1706.03762))
- Provide an implementation of gradient accumulation

### Data

- Accepted vocab formats: .pickle
- Accepted data formats: tokenized and on sub-word level 

- Batch size given in number of tokens without padding
- The batch size is never exceeded
- Data is sorted for train, search and eval tasks and re-ordered when outputed
- In training the data is sorted, re-shuffled and batched using a bucketing algorithm
- Data is read in the background reducing data loading time to 0


### Scores

- Cross-entropy and perplexity implementation
- Scores to check the batch redundancy and fullness

### Search

- Search algorithms: Greedy and beam search

### Timer

- To enable timing, set the global time flag with `Globals.set_time_flag(True)`
- Tensorflow will be started in eager mode and the flag `tf.config.experimental.set_synchronous_execution(True)` is set to ensure that functions are not interleaved
- There are two timer objects which deliver timing functionality: `Timer` and `ContextTimer`.
- The `Timer` is used to measure modules inside the model wheras the `ContextTimer` is used for all other functions.
- `Timer`: wraps the `__call__` function of a tf.Module and stores the timings in a tf.Variable which is a non-trainable variable of the model. The variable is identified by the name of the tf.Module.
- `ContextTimer`: Has nothing to do with tf. It measures the scope in which it is created. The result is stored in a static member variable and identified by the name.

## Setup ##

We use poetry to manage the python environment and manage dependencies.
Hence, a list of python dependencies can be found in `pyproject.toml`.
Since we had trouble installing Horovod with poetry, it must be installed manually for now.
This is only temporary and must be fixed in the future.

Except from the python dependencies we recommend to use cuda-11.3 and nccl-2.14.
On the AppTek cluster installations for these can be found in /home/fschmidt/lib. 
Before installing Horovod, the corresponding path variables must be set which is done in `/home/fschmidt/code/workflowManager/bash/setup_cuda.bash`.

### Install Poetry ####

```bash
curl -sSL https://install.python-poetry.org | POETRY_HOME=YOUR_POETRY_DIR python3 -
```

Add poetry to your PATH variable:
```bash
export YOUR_POETRY_DIR:$PATH
```

### Install Python Dependencies except Horovod ####

With the following steps poetry will create a virtualenv, by default in ~/.cache/pypoetry/virtualenvs.
From the root folder call:
```bash
poetry shell
source /home/fschmidt/code/workflowManager/bash/setup_cuda.bash
poetry install
```

If you manually want to select the virtualenv, run the following commands with your paths from our root folder.
```bash
python3 -m venv YOUR_VENV_FOLDER
source YOUR_VENV_FOLDER/bin/activate
pip3 install --upgrade pip3
source /home/fschmidt/code/workflowManager/bash/setup_cuda.bash
poetry install
```

### Install Horovod ####

Enter the virtualenv created in the step before and run the following commands. 

```bash
source /home/fschmidt/code/workflowManager/bash/setup_cuda.bash
HOROVOD_WITH_TENSORFLOW=1 pip3 install --no-cache-dir horovod
```

### Add tensorflowSeq2Seq to your PYTHONPATH ###

```bash
export PYTHONPATH=$PYTHONPATH:YOUR_PATH/tensorflowSeq2Seq
```

## Run ##

To this point, we provide the two main functionalities for machine translation models: training and search.
Both have their entry point in tensorflowSeq2Seq/train.py and tensorflowSeq2Seq/search.py.
Since we use argparse you can view a most up-to-date version of the parameters with
```bash
tensorflowSeq2Seq/train.py --help
tensorflowSeq2Seq/search.py --help
```

### Train ###

Usage of train.py:

```bash
usage: train.py [-h] --config CONFIG --output-folder OUTPUT_FOLDER [--resume-training RESUME_TRAINING] [--resume-training-from RESUME_TRAINING_FROM] [--number-of-gpus NUMBER_OF_GPUS]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       The path to the config.yaml which contains all user defined parameters.
  --output-folder OUTPUT_FOLDER
                        The folder in which to write the training output (ckpts, learning-rates, perplexities etc.)
  --resume-training RESUME_TRAINING
                        If you want to resume a training, set this flag to 1 and specify the directory with "resume-training-from".
  --resume-training-from RESUME_TRAINING_FROM
                        If you want to resume a training, specify the output directory here. We expect it to have the same layout as a newly created one.
  --number-of-gpus NUMBER_OF_GPUS
                        This is usually specified in the config but can also be overwritten from the cli.
```

### Search ###

Usage of search.py:

```bash
usage: search.py [-h] --config CONFIG --checkpoint-prefix CHECKPOINT_PREFIX [--output-folder OUTPUT_FOLDER] [--number-of-gpus NUMBER_OF_GPUS]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       The path to the config.yaml which contains all user defined parameters. It may or may not match the one trained with. This is up to the user to ensure.
  --checkpoint-prefix CHECKPOINT_PREFIX
                        The checkpoint prefix pointing to the model weights.
  --output-folder OUTPUT_FOLDER
                        The output folder in which to write the score and hypotheses.
  --number-of-gpus NUMBER_OF_GPUS
                        This is usually specified in the config but can also be overwritten from the cli. However, in search this can only be 0 or 1. We do not support multi-gpu decoding.
                        If you set it to >1 we will set it back to 1 so that you dont need to modify the config in search.
```
