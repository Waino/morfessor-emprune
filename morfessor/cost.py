
import logging
import numbers


from .corpus import CorpusEncoding, LexiconEncoding, AnnotatedCorpusEncoding,FixedCorpusWeight

_logger = logging.getLogger(__name__)


class Cost(object):
    """Class for calculating the entropy (encoding length) of a corpus and lexicon.

    """
    def __init__(self, corpusweight=1.0):
        # Cost variables
        self._lexicon_coding = LexiconEncoding()
        self._corpus_coding = CorpusEncoding(self._lexicon_coding)
        self._annot_coding = None

        self._corpus_weight_updater = None
        
        #Set corpus weight updater
        self.set_corpus_weight_updater(corpusweight)

    def set_corpus_weight_updater(self, corpus_weight):
        if corpus_weight is None:
            self._corpus_weight_updater = FixedCorpusWeight(1.0)
        elif isinstance(corpus_weight, numbers.Number):
            self._corpus_weight_updater = FixedCorpusWeight(corpus_weight)
        else:
            self._corpus_weight_updater = corpus_weight

        self._corpus_weight_updater.update(self, 0)

    def increase_weight(self):
        pass

    def decrease_weight(self):
        pass

    def cost(self):
        pass

    def update(self, construction, delta):
        pass

    def coding_length(self, construction):
        pass

