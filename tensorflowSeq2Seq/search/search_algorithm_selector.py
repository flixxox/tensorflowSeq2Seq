
from tensorflowSeq2Seq.search.greedy_search import GreedySearch
from tensorflowSeq2Seq.search.beam_search_fast import BeamSearchFast
from tensorflowSeq2Seq.search.beam_search_long import BeamSearchLong

class SearchAlgorithmSelector:

    def __init__(self):
        pass

    @staticmethod
    def create_search_algorithm_from_config(config, model, vocab_src, vocab_tgt):
        
        if config['search_algorithm'] == 'Greedy':
            
            return GreedySearch.create_search_algorithm_from_config(config, model, vocab_src, vocab_tgt)

        elif config['search_algorithm'] == 'BeamLong':
            
            return BeamSearchLong.create_search_algorithm_from_config(config, model, vocab_src, vocab_tgt)

        elif config['search_algorithm'] == 'BeamFast':
            
            return BeamSearchFast.create_search_algorithm_from_config(config, model, vocab_src, vocab_tgt)
        
        else:

            assert True == False, f'Unrecognized search algorithm {config["search_algorithm"]}'