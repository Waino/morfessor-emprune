
import logging
import numbers
from collections import Counter

import math

from .corpus import CorpusEncoding, LexiconEncoding, AnnotatedCorpusEncoding,FixedCorpusWeight

_logger = logging.getLogger(__name__)


class Cost(object):
    """Class for calculating the entropy (encoding length) of a corpus and lexicon.

    """
    def __init__(self, contr_class, corpusweight=1.0):
        self.cc = contr_class
        # Cost variables
        self._lexicon_coding = LexiconEncoding()
        self._corpus_coding = CorpusEncoding(self._lexicon_coding)
        self._annot_coding = None

        self._corpus_weight_updater = None

        #Set corpus weight updater
        self.set_corpus_weight_updater(corpusweight)

        self.counts = Counter()

    def set_corpus_weight_updater(self, corpus_weight):
        if corpus_weight is None:
            self._corpus_weight_updater = FixedCorpusWeight(1.0)
        elif isinstance(corpus_weight, numbers.Number):
            self._corpus_weight_updater = FixedCorpusWeight(corpus_weight)
        else:
            self._corpus_weight_updater = corpus_weight

        self._corpus_weight_updater.update(self, 0)

    def set_corpus_coding_weight(self, weight):
        self._corpus_coding.weight = weight

    def cost(self):
        return self._lexicon_coding.get_cost() + self._corpus_coding.get_cost()

    def update(self, construction, delta):
        if delta == 0:
            return

        if self.counts[construction] == 0:
            self._lexicon_coding.add(self.cc.lex_key(construction))

        old_count = self.counts[construction]
        self.counts[construction] += delta

        self._corpus_coding.update_count(self.cc.corpus_key(construction), old_count, self.counts[construction])

        if self.counts[construction] == 0:
            self._lexicon_coding.remove(self.cc.lex_key(construction))

    def coding_length(self, construction):
        pass

    def tokens(self):
        pass

    def all_tokens(self):
        return self._corpus_coding.tokens + self._corpus_coding.boundaries

    def newbound_cost(self, count):
        cost = (self._lexicon_coding.boundaries + count) * math.log(self._lexicon_coding.boundaries + count)
        if self._lexicon_coding.boundaries > 0:
            cost -= self._lexicon_coding.boundaries * math.log(self._lexicon_coding.boundaries)
        return cost / self._corpus_coding.weight

    def bad_likelihood(self, compound, addcount):
        lt = math.log(self.all_tokens() + addcount) if addcount > 0 else 0
        nb = self.newbound_cost(addcount) if addcount > 0 else 0

        return 1.0 + len(self.cc.corpus_key(compound)) * lt + nb + \
                        self._lexicon_coding.get_codelength(compound) / \
                        self._corpus_coding.weight

    def types(self):
        pass
