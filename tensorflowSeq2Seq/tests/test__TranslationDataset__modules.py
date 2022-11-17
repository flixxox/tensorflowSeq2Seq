import unittest

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflowSeq2Seq.util.config import Config
from tensorflowSeq2Seq.util.debug import my_print
from tensorflowSeq2Seq.util.globals import Globals
from tensorflowSeq2Seq.util.data import Vocabulary, TranslationDataset

class TestData(unittest.TestCase):

    def test__load_data_ptrs(self):
        
        dataset, (_, data_tgt), _, _ = self.__prepare_objects()

        dataset.load_data_ptrs()

        exp_data_ptrs = self.__get_hand_crafted_data_ptrs(data_tgt)

        for i in range(len(exp_data_ptrs)):

            assert dataset.data_ptrs[0][i] == exp_data_ptrs[i]

    def __get_hand_crafted_data_ptrs(self, data_tgt):

        data_ptrs = []

        for i, tgt in enumerate(data_tgt):
            data_ptrs.append((i, len(tgt)+1))

        return data_ptrs


    def test__apply_epoch_split(self):
        
        self.__test_one_epoch_split(2)
        self.__test_one_epoch_split(5)
        self.__test_one_epoch_split(9)
        self.__test_one_epoch_split(20)
        self.__test_one_epoch_split(50)

    def __test_one_epoch_split(self, epoch_split):

        dataset, (_, data_tgt), _, _ = self.__prepare_objects(epoch_split=epoch_split)

        dataset.load_data_ptrs()

        exp_data_ptrs = dataset.data_ptrs[0].copy()

        dataset.apply_epoch_split()

        assert len(dataset.data_ptrs) == epoch_split

        offset = len(data_tgt) // epoch_split

        assert sum([len(x) for x in dataset.data_ptrs]) == len(exp_data_ptrs)

        for i in range(epoch_split):
            for j in range(len(dataset.data_ptrs[i])):

                assert dataset.data_ptrs[i][j] == exp_data_ptrs[(i*offset) + j]


    def test__assign_to_worker(self):
        
        self.__test_one_worker_setup(4, 2)
        self.__test_one_worker_setup(4, 3)
        self.__test_one_worker_setup(4, 4)
        self.__test_one_worker_setup(4, 5)

        self.__test_one_worker_setup(5, 2)
        self.__test_one_worker_setup(5, 3)
        self.__test_one_worker_setup(5, 4)
        self.__test_one_worker_setup(5, 5)

    def __test_one_worker_setup(self, epoch_split, workers):

        dataset, (_, data_tgt), _, _ = self.__prepare_objects(epoch_split=epoch_split, workers=workers)

        for rank in range(workers):

            dataset.data_ptrs = [[]]

            dataset.load_data_ptrs()

            dataset.apply_epoch_split()

            exp_data_ptrs = dataset.data_ptrs.copy()

            dataset.assign_to_worker(rank)

            for i in range(epoch_split):
                for j in range(len(dataset.data_ptrs[i])):

                    assert dataset.data_ptrs[i][j] == exp_data_ptrs[i][(j*workers) + rank]


    def test__load_data_to_memory(self):
        
        dataset, (data_src, data_tgt), vocab_src, vocab_tgt = self.__prepare_objects()

        data_src = [vocab_src.tokenize(s) for s in data_src]
        data_tgt = [vocab_tgt.tokenize(t) for t in data_tgt]

        dataset.data_ptrs = [[
                (0, 0),
                (10, 0),
                (20, 0),
                (30, 0),
                (49, 0),
            ],
            [
                (1,0),
                (11,0),
                (21,0),
                (31,0)
            ]
        ]

        self.__test_loading_for_data_ptrs(dataset, data_src, data_tgt)

    def __test_loading_for_data_ptrs(self, dataset, data_src, data_tgt):

        dataset.load_data_to_memory()

        for i in range(len(dataset.data_ptrs)):
            for j in range(len(dataset.data_ptrs[i])):
                
                ptr = dataset.data_ptrs[i][j]

                src = dataset.data[ptr[0]][0]
                tgt = dataset.data[ptr[0]][1]

                assert len(src) > 0
                assert len(tgt) > 0

                assert src == data_src[ptr[0]]
                assert tgt == data_tgt[ptr[0]]


    def test__ptrs_to_tensor(self):
        
        dataset, (data_src, data_tgt), vocab_src, vocab_tgt = self.__prepare_objects()

        dataset.load_data_ptrs()

        dataset.load_data_to_memory()

        ptrs = [
            0,
            1,
            4,
            10,
            20,
            30,
            44,
        ]

        (src, tgt, out) = dataset.ptrs_to_tensor(ptrs)

        src = list(src.numpy())
        tgt = list(tgt.numpy())
        out = list(out.numpy())

        for i in range(len(ptrs)):

            s = list(src[i])
            t = list(tgt[i])
            o = list(out[i])

            s = vocab_src.detokenize(s)
            s = vocab_src.remove_padding(s)

            t = vocab_tgt.detokenize(t)
            t = vocab_tgt.remove_padding(t)

            o = vocab_tgt.detokenize(o)
            o = vocab_tgt.remove_padding(o)

            s = s[1:-1]
            t = t[1:]
            o = o[:-1]

            exp_src = data_src[ptrs[i]]
            exp_tgt = data_tgt[ptrs[i]]

            assert exp_src == s
            assert exp_tgt == t == o


    def __prepare_objects(self, epoch_split=1, workers=1, in_memory=True):

        hvd.init()

        path_test_vocab_src = 'tests/res/mockup_vocab_src.pickle'
        path_test_vocab_tgt = 'tests/res/mockup_vocab_tgt.pickle'
        path_test_data_src  = 'tests/res/mockup_data_src'
        path_test_data_tgt  = 'tests/res/mockup_data_tgt'

        Globals.set_number_of_workers(workers, force=True)
        Globals.set_train_flag(True)

        vocab_src = Vocabulary.create_vocab(path_test_vocab_src)
        vocab_tgt = Vocabulary.create_vocab(path_test_vocab_tgt)

        dataset = TranslationDataset(
            src_path    = path_test_data_src,
            tgt_path    = path_test_data_tgt, 
            vocab_src   = vocab_src,
            vocab_tgt   = vocab_tgt,
            name        = 'test_dataset',
            maxI        = 256,
            epoch_split = epoch_split,
            in_memory   = in_memory
        )

        return dataset, self.__read_test_data(path_test_data_src, path_test_data_tgt), vocab_src, vocab_tgt

    def __read_test_data(self, test_data_src, test_data_tgt):

        src_data = []
        tgt_data = []

        with open(test_data_src, "r") as src_file, open(test_data_tgt, "r") as tgt_file:
            
            for (src, tgt) in zip(src_file, tgt_file):

                src = src.strip().replace("\n", "").split(" ")
                tgt = tgt.strip().replace("\n", "").split(" ")

                src_data.append(src)
                tgt_data.append(tgt)

        return src_data, tgt_data


