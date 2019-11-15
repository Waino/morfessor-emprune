from __future__ import unicode_literals
import collections
import heapq
import itertools
import logging
import math
import numbers
import random
import sys

from scipy.special import digamma
# import math
# def digamma(x):
#   result = 0.0
#   while x < 7:
#     result -= 1 / x
#     x += 1
#   x -= 1.0 / 2.0
#   xx = 1.0 / x
#   xx2 = xx * xx
#   xx4 = xx2 * xx2
#   result += (math.log(x) + (1.0 / 24.0) * xx2 - (7.0 / 960.0) * xx4 +
#              (31.0 / 8064.0) * xx4 * xx2 - (127.0 / 30720.0) * xx4 * xx4)
#   return result

from .cost import Cost, EmCost
from .constructions.base import BaseConstructionMethods
from .corpus import LexiconEncoding, CorpusEncoding, \
    AnnotatedCorpusEncoding, FixedCorpusWeight
from .utils import _progress, tail, logsumexp, categorical
from .exception import MorfessorException, SegmentOnlyModelException

_logger = logging.getLogger(__name__)


# rcount = root count (from corpus)
# count = total count of the node
# splitloc = integer or tuple. Location(s) of the possible splits for virtual
#            constructions; empty tuple or 0 if real construction
ConstrNode = collections.namedtuple('ConstrNode',
                                    ['rcount', 'count', 'splitloc'])

MODE_NORMAL = 'normal'
MODE_EM = 'em'
MODE_SEGMENT_ONLY = 'segment_only'

EPS = 1e-8

PRUNE_ALWAYS = 0
PRUNE_GAIN = 1
PRUNE_LOSS = 2
PRUNE_NEVER_DBL = 3
PRUNE_NEVER_NO_ALT = 4
PRUNE_NEVER_CHAR = 5

PruneStats = collections.namedtuple('PruneStats',
    ['construction', 'threshold_alpha', 'delta_lc', 'delta_cc', 'delta_cost', 'decision'])


class BaselineModel(object):
    """Morfessor Baseline model class.

    Implements training of and segmenting with a Morfessor model. The model
    is complete agnostic to whether it is used with lists of strings (finding
    phrases in sentences) or strings of characters (finding morphs in words).

    """

    penalty = -9999.9

    def __init__(self, corpusweight=None, use_skips=False, constr_class=None,
                 use_em=False, em_substr=None, nolexcost=False, freq_distr='zero'):
        """Initialize a new model instance.

        Arguments:
            forcesplit_list: force segmentations on the characters in
                               the given list
            corpusweight: weight for the corpus cost
            use_skips: randomly skip frequently occurring constructions
                         to speed up training
            nosplit_re: regular expression string for preventing splitting
                          in certain contexts
            use_em: use em+prune training
            em_substr: substring lexicon
            nolexcost: ignore lexicon cost with EM+prune

        """

        self.cc = constr_class if constr_class is not None else BaseConstructionMethods()

        # In analyses for each construction a ConstrNode is stored. All
        # training data has a rcount (real count) > 0. All real morphemes
        # have no split locations.
        self._analyses = {}

        # Flag to indicate the mode in which the model is operating
        self._mode = MODE_EM if use_em else MODE_NORMAL

        # Cost variables
        # self._lexicon_coding = LexiconEncoding()
        # self._corpus_coding = CorpusEncoding(self._lexicon_coding)
        # self._annot_coding = None

        if use_em:
            self.cost = EmCost(self.cc, corpusweight, nolexcost, freq_distr)
            self.cost.load_lexicon(em_substr)
            self.em_autotune_alpha = False  # overridden later
        else:
            self.cost = Cost(self.cc, corpusweight)

        #Set corpus weight updater
        self.set_corpus_weight_updater(corpusweight)

    def set_corpus_weight_updater(self, corpus_weight):
        if corpus_weight is None:
            self._corpus_weight_updater = FixedCorpusWeight(1.0)
        elif isinstance(corpus_weight, numbers.Number):
            self._corpus_weight_updater = FixedCorpusWeight(corpus_weight)
        else:
            self._corpus_weight_updater = corpus_weight

    @property
    def _segment_only(self):
        return self._mode == MODE_SEGMENT_ONLY

    @property
    def tokens(self):
        """Return the number of construction tokens."""
        return self.cost.tokens()

    @property
    def types(self):
        """Return the number of construction types."""
        return self.cost.types() - 1  # do not include boundary

    def _check_segment_only(self):
        if self._segment_only:
            raise SegmentOnlyModelException()

    def _check_normal_mode(self):
        if not self._mode == MODE_NORMAL:
            raise Exception('Model must be in normal mode')

    def _epoch_checks(self):
        """Apply per epoch checks"""
        # self._check_integrity()
        pass

    def _epoch_update(self, epoch_num):
        """Do model updates that are necessary between training epochs.

        The argument is the number of training epochs finished.

        In practice, this does two things:
        - If random skipping is in use, reset construction counters.
        - If semi-supervised learning is in use and there are alternative
          analyses in the annotated data, select the annotations that are
          most likely given the model parameters. If not hand-set, update
          the weight of the annotated corpus.

        This method should also be run prior to training (with the
        epoch number argument as 0).

        """
        forced_epochs = 0
        if self._corpus_weight_updater is not None:
            if self._corpus_weight_updater.update(self, epoch_num):
                forced_epochs += 2

        # if self._use_skips:
        #     self._counter = collections.Counter()
        # if self._supervised:
        #     self._update_annotation_choices()
        #     self._annot_coding.update_weight()

        return forced_epochs

    def _update_annotation_choices(self):
        """Update the selection of alternative analyses in annotations.

        For semi-supervised models, select the most likely alternative
        analyses included in the annotations of the compounds.

        """
        if not self._supervised:
            return

        # Collect constructions from the most probable segmentations
        # and add missing compounds also to the unannotated data
        constructions = collections.Counter()
        for compound, alternatives in self.annotations.items():
            if not compound in self._analyses:
                self._add_compound(compound, 1)

            analysis, cost = self._best_analysis(alternatives)
            for m in analysis:
                constructions[m] += self._analyses[compound].rcount

        # Apply the selected constructions in annotated corpus coding
        self._annot_coding.set_constructions(constructions)
        for constr in constructions.keys():
            count = self.get_construction_count(constr)
            self._annot_coding.set_count(constr, count)

    def _best_analysis(self, choices):
        """Select the best analysis out of the given choices."""
        bestcost = None
        bestanalysis = None
        for analysis in choices:
            cost = 0.0
            for constr in analysis:
                count = self.get_construction_count(constr)
                if count > 0:
                    cost += (math.log(self.cost.tokens()) -
                             math.log(count))
                else:
                    cost -= self.penalty  # penalty is negative
            if bestcost is None or cost < bestcost:
                bestcost = cost
                bestanalysis = analysis
        return bestanalysis, bestcost

    def _add_compound(self, compound, c):
        """Add compound with count c to data."""
        self.cost.update_boundaries(compound, c)
        if self._mode == MODE_NORMAL:
            self._modify_construction_count(compound, c)
            oldrc = self._analyses[compound].rcount
            self._analyses[compound] = \
                self._analyses[compound]._replace(rcount=oldrc + c)
        else:
            self._analyses[compound] = ConstrNode(c, c, [])

    def _remove(self, construction):
        """Remove construction from model."""
        self._check_normal_mode()
        rcount, count, splitloc = self._analyses[construction]
        self._modify_construction_count(construction, -count)
        return rcount, count

    def _clear_compound_analysis(self, compound):
        """Clear analysis of a compound from model"""
        pass

    def _set_compound_analysis(self, compound, parts):
        """Set analysis of compound to according to given segmentation.

        Arguments:
            compound: compound to split
            parts: desired constructions of the compound

        """
        self._check_normal_mode()
        parts = list(parts)
        if len(parts) == 1:
            rcount, count = self._remove(compound)
            self._analyses[compound] = ConstrNode(rcount, 0, tuple())
            self._modify_construction_count(compound, count)
        else:
            rcount, count = self._remove(compound)

            splitloc = tuple(self.cc.parts_to_splitlocs(parts))
            self._analyses[compound] = ConstrNode(rcount, count, splitloc)
            for constr in parts:
                self._modify_construction_count(constr, count)

    def get_construction_count(self, construction):
        """Return (real) count of the construction."""
        return self.cost.counts.get(construction, 0)

    def _test_skip(self, construction):
        """Return true if construction should be skipped."""
        if construction in self._counter:
            t = self._counter[construction]
            if random.random() > 1.0 / max(1, t):
                return True
        self._counter[construction] += 1
        return False

    def _viterbi_optimize(self, compound, addcount=0, maxlen=30):
        """Optimize segmentation of the compound using the Viterbi algorithm.

        Arguments:
          compound: compound to optimize
          addcount: constant for additive smoothing of Viterbi probs
          maxlen: maximum length for a construction

        Returns list of segments.

        """
        self._check_normal_mode()
        if self._use_skips and self._test_skip(compound):
            return self.segment(compound)

        # Use Viterbi algorithm to optimize the subsegments
        constructions = []
        for part in self.cc.splitn(compound, self.cc.force_split_locations(compound)):
            constructions.extend(self.viterbi_segment(part, addcount=addcount,
                                                  maxlen=maxlen)[0])
        self._set_compound_analysis(compound, constructions)
        return constructions

    def _recursive_optimize(self, compound):
        """Optimize segmentation of the compound using recursive splitting.

        Returns list of segments.

        """
        self._check_normal_mode()
        # if self._use_skips and self._test_skip(compound):
        #     return self.segment(compound)
        # Collect forced subsegments

        parts = list(self.cc.splitn(compound, self.cc.force_split_locations(compound)))
        if len(parts) == 1:
            # just one part
            return self._recursive_split(compound)
        self._set_compound_analysis(compound, parts)
        # Use recursive algorithm to optimize the subsegments
        constructions = []
        for part in parts:
            constructions += self._recursive_split(part)
        return constructions

    def _recursive_split(self, construction):
        """Optimize segmentation of the construction by recursive splitting.

        Returns list of segments.

        """
        # if self._use_skips and self._test_skip(construction):
        #     return self.segment(construction)
        rcount, count = self._remove(construction)

        # Check all binary splits and no split
        self._modify_construction_count(construction, count)
        mincost = self.get_cost()
        self._modify_construction_count(construction, -count)

        best_splitloc = None

        for loc in self.cc.split_locations(construction):
            prefix, suffix = self.cc.split(construction, loc)
            self._modify_construction_count(prefix, count)
            self._modify_construction_count(suffix, count)
            cost = self.get_cost()
            self._modify_construction_count(prefix, -count)
            self._modify_construction_count(suffix, -count)
            if cost <= mincost:
                mincost = cost
                best_splitloc = loc

        if best_splitloc:
            # Virtual construction
            self._analyses[construction] = ConstrNode(rcount, count, best_splitloc)
            prefix, suffix = self.cc.split(construction, best_splitloc)
            self._modify_construction_count(prefix, count)
            self._modify_construction_count(suffix, count)
            lp = self._recursive_split(prefix)
            if suffix != prefix:
                return lp + self._recursive_split(suffix)
            else:
                return lp + lp
        else:
            # Real construction
            self._analyses[construction] = ConstrNode(rcount, 0, None)
            self._modify_construction_count(construction, count)
            return [construction]

    def _modify_construction_count(self, construction, dcount):
        """Modify the count of construction by dcount.

        For virtual constructions, recurses to child nodes in the
        tree. For real constructions, adds/removes construction
        to/from the lexicon whenever necessary.

        """
        if dcount == 0 or construction is None:
            return
        if construction in self._analyses:
            rcount, count, splitloc = self._analyses[construction]
        else:
            rcount, count, splitloc = 0, 0, None
        newcount = count + dcount
        # observe that this comparison will not work correctly if counts
        # are floats rather than ints
        if newcount == 0:
            if construction in self._analyses:
                del self._analyses[construction]
        else:
            self._analyses[construction] = ConstrNode(rcount, newcount,
                                                      splitloc)
        if splitloc:
            # Virtual construction
            for child in self.cc.splitn(construction, splitloc):
                self._modify_construction_count(child, dcount)
        else:
            self.cost.update(construction, newcount-count)
            # Real construction

    def get_compounds(self):
        """Return the compound types stored by the model."""
        self._check_segment_only()
        return [w for w, node in self._analyses.items()
                if node.rcount > 0]

    def get_compound_counts(self):
        """Return the compound types stored by the model."""
        self._check_segment_only()
        return [(w, node.rcount) for (w, node) in self._analyses.items()
                if node.rcount > 0]

    def get_constructions(self):
        """Return a list of the present constructions and their counts."""
        return sorted((c, node.count) for c, node in self._analyses.items()
                      if not node.splitloc)

    def get_cost(self):
        """Return current model encoding cost."""
        return self.cost.cost()
        cost = self.cost.cost()
        if self._supervised:
            return cost + self._annot_coding.get_cost()
        else:
            return cost

    def get_segmentations(self):
        """Retrieve segmentations for all compounds encoded by the model."""
        self._check_normal_mode()
        for w in sorted(self._analyses.keys()):
            c = self._analyses[w].rcount
            if c > 0:
                yield c, w, self.segment(w)

    def get_pseudomodel(self, viterbismooth, viterbimaxlen):
        self._check_segment_only()
        for w in sorted(self._analyses.keys()):
            node = self._analyses[w]
            if node.rcount == 0:
                continue
            constructions, _ = self.viterbi_segment(
                w, viterbismooth, viterbimaxlen)
            yield (node.rcount, w, constructions)

    def load_data(self, data):
        """Load data to initialize the model for batch training.

        Arguments:
            data: iterator of DataPoint tuples

        Adds the compounds in the corpus to the model lexicon. Returns
        the total cost.

        """
        self._check_segment_only()
        for dp in data:
            self._add_compound(dp.compound, dp.count)
            if self._mode == MODE_NORMAL:
                self._clear_compound_analysis(dp.compound)
                self._set_compound_analysis(dp.compound, self.cc.splitn(dp.compound, dp.splitlocs))
        return self.get_cost()

    # FIXME: refactor?
    def load_segmentations(self, segmentations):
        self._check_normal_mode()
        for count, compound, constructions in segmentations:
            splitlocs = tuple(self.cc.parts_to_splitlocs(constructions))
            self._add_compound(compound, count)
            self._clear_compound_analysis(compound)
            self._set_compound_analysis(compound, self.cc.splitn(compound, splitlocs))
        return self.get_cost()

    def segment(self, compound):
        """Segment the compound by looking it up in the model analyses.

        Raises KeyError if compound is not present in the training
        data. For segmenting new words, use viterbi_segment(compound).

        """
        self._check_normal_mode()
        _, _, splitloc = self._analyses[compound]
        constructions = []
        if splitloc:
            for part in self.cc.splitn(compound, splitloc):
                constructions += self.segment(part)
        else:
            constructions.append(compound)

        return constructions

    def e_step(self, maxlen):
        expected = collections.Counter()
        compounds = list(self.get_compound_counts())
        tot_cost = 0
        for compound, freq in compounds:
            w_expected, cost = self.forward_backward(compound, freq, maxlen)
            expected.update(w_expected)
            tot_cost += cost
        return expected, tot_cost

    def e_step_hard(self, maxlen):
        expected = collections.Counter()
        compounds = list(self.get_compound_counts())
        tot_cost = 0
        for compound, freq in compounds:
            constructions, cost = self.viterbi_segment(
                compound, addcount=0.0, maxlen=maxlen)
            for cons in constructions:
                expected[cons] += freq
            tot_cost += cost
        return expected, tot_cost

    def m_step(self, expected, expected_freq_threshold):
        # prune out infrequent
        # FIXME: is protecting length 1 useful? max(c, 1e-6))?
        expected = collections.Counter(
            dict((w, c) for (w, c) in expected.items()
                 if c > expected_freq_threshold or len(w) == 1))

        # apply exp digamma for Bayesianified/DPified EM
        # acts as a sparse prior
        # https://cs.stanford.edu/~pliang/papers/tutorial-acl2007-talk.pdf
        tot = sum(expected.values())
        multiplier = tot / math.exp(digamma(tot))
        for construction in expected.keys():
            expected[construction] = math.exp(digamma(expected[construction])) * multiplier

        # set model parameters
        self.cost.counts = expected

    def prune_lexicon(self, prune_criterion):
        self.cost.reset()
        prune_stats = list(self.compute_prune_stats())
        pruned, done = prune_criterion(prune_stats)
        n_pruned = len(pruned)
        for construction in pruned:
            # prune out selected constructions
            count = self.cost.counts[construction]
            self.cost.update(construction, -count)
            del self.cost.counts[construction]
        return self.get_cost(), done

    def compute_prune_stats(self):
        orig_lc, orig_cc = self.cost.cost_before_tuning()
        constructions = list(w for w, c in self.cost.counts.most_common())
        current_alpha = self.get_corpus_coding_weight()
        for construction in constructions:
            if len(construction) == 1:
                yield PruneStats(construction, -math.inf, 0, 0, 0, PRUNE_NEVER_CHAR)
                continue
            # assume all probability mass goes to viterbi segmentation
            replacement, _ = self.viterbi_segment(
                construction, taboo=[construction], addcount=0)
            if replacement == construction:
                yield PruneStats(construction, -math.inf, 0, 0, 0, PRUNE_NEVER_NO_ALT)
                continue
            count = self.cost.counts[construction]

            # apply change
            self.cost.update(construction, -count)
            for replcons in replacement:
                self.cost.update(replcons, count)
            lc, cc = self.cost.cost_before_tuning()
            # revert change
            self.cost.update(construction, count)
            for replcons in replacement:
                self.cost.update(replcons, -count)

            delta_lc = lc - orig_lc
            delta_cc = cc - orig_cc
            threshold_alpha, delta_cost, decision = self.prune_cost_at_alpha(
                current_alpha, delta_lc, delta_cc)
            yield PruneStats(construction,
                             threshold_alpha,
                             delta_lc, delta_cc,
                             delta_cost, decision)

    def prune_cost_at_alpha(self, alpha, delta_lc, delta_cc):
        delta_cost = delta_lc + (alpha * delta_cc)
        # tuning can't affect if both deltas have the same sign
        if delta_lc < 0 and delta_cc < 0:
            decision = PRUNE_ALWAYS
            threshold_alpha = math.inf
        elif delta_lc > 0 and delta_cc > 0:
            decision = PRUNE_NEVER_DBL
            threshold_alpha = -math.inf
        else:
            # if deltas have opposite sign,
            # compute the threshold alpha for which they cancel out
            threshold_alpha = abs(delta_lc / (delta_cc + EPS))
            # decicion based on current alpha
            decision = PRUNE_GAIN if delta_cost < 0 else PRUNE_LOSS
        return threshold_alpha, delta_cost, decision

    def reweight_prune_stats(self, prune_stats, optimal_alpha):
        for stat in prune_stats:
            threshold_alpha, delta_cost, decision = self.prune_cost_at_alpha(
                optimal_alpha, stat.delta_lc, stat.delta_cc)
            yield PruneStats(stat.construction,
                             stat.threshold_alpha,
                             stat.delta_lc, stat.delta_cc,
                             delta_cost, decision)

    def prune_criterion_lexicon_size(self, proportion, goal_lexicon):
        # prune at most proportion. prune until goal_lexicon is reached
        def prune_criterion(prune_stats):
            n_tot = len(prune_stats)
            max_prune_prop = int(math.ceil(n_tot * proportion))
            max_prune_goal = max(0, int(n_tot - goal_lexicon))
            max_prune = min(max_prune_prop, max_prune_goal)
            # done unless epoch quota was the stopping reason
            done = max_prune_goal <= max_prune_prop
            prune_stats.sort(key=lambda x: (x.decision, x.delta_cost))
            pruned = [x.construction for x in prune_stats]
            return pruned, done
        return prune_criterion

    def prune_criterion_mdl(self, proportion):
        # prune at most proportion. prune based on decision
        def prune_criterion(prune_stats):
            n_tot = len(prune_stats)
            max_prune_prop = int(math.ceil(n_tot * proportion))
            prune_stats.sort(key=lambda x: (x.decision, x.delta_cost))
            pruned = []
            for (i, stat) in enumerate(prune_stats):
                if i >= max_prune_prop:
                    return pruned, False
                if stat.decision >= PRUNE_LOSS:
                    return pruned, True
                pruned.append(stat.construction)
            # pruned everything
            _logger.info('pruned everything!')
            return pruned, True
        return prune_criterion

    def prune_criterion_autotune(self, proportion, goal_lexicon):
        # determine optimal alpha. prune at most proportion. prune based on decision
        def prune_criterion(prune_stats):
            # determine optimal alpha
            prune_stats.sort(key=lambda x: (x.decision, -x.threshold_alpha))
            optimal_alpha = prune_stats[-int(goal_lexicon)].threshold_alpha
            if optimal_alpha == -math.inf:
                _logger.info('cannot reach goal lexicon by tuning: too many always keep')
                optimal_alpha = min(x.threshold_alpha for x in prune_stats
                                    if x.decision in (PRUNE_GAIN, PRUNE_LOSS))
            _logger.info("Corpus weight set to {}".format(optimal_alpha))
            self.set_corpus_coding_weight(optimal_alpha)
            prune_stats = list(self.reweight_prune_stats(prune_stats, optimal_alpha))

            # continue with pruning
            n_tot = len(prune_stats)
            max_prune_prop = int(math.ceil(n_tot * proportion))
            max_prune_goal = max(0, int(n_tot - goal_lexicon))
            max_prune = min(max_prune_prop, max_prune_goal)
            # done unless epoch quota was the stopping reason
            done = max_prune_goal <= max_prune_prop
            prune_stats.sort(key=lambda x: (x.decision, x.delta_cost))
            pruned = []
            for (i, stat) in enumerate(prune_stats):
                if i >= max_prune:
                    return pruned, done
                if stat.decision >= PRUNE_LOSS:
                    return pruned, True
                pruned.append(stat.construction)
            # pruned everything
            _logger.info('pruned everything!')
            return pruned, True
        return prune_criterion

    def train_em_prune(self, prune_criterion,
                       max_epochs=5, sub_epochs=3,
                       expected_freq_threshold=0.5,
                       maxlen=30, use_lateen=False):
        done = False
        for epoch in range(max_epochs):
            for sub_epoch in range(sub_epochs):
                # E-step
                if use_lateen and sub_epoch == sub_epochs - 1:
                    _logger.info('Lateen EM: using Viterbi e-step')
                    expected, cost = self.e_step_hard(maxlen=maxlen)
                else:
                    expected, cost = self.e_step(maxlen=maxlen)
                _logger.info("E-step cost: %s tokens: %s" % (cost, self.cost.all_tokens()))
                # M-step
                self.m_step(
                    expected,
                    expected_freq_threshold=expected_freq_threshold)
            if done:
                break
            # cost-based pruning of lexicon
            cost, done = self.prune_lexicon(prune_criterion)
            lc, cc = self.cost.cost_before_tuning()
            _logger.info("Cost after pruning: %s types: %s tokens: %s" %
                (cost, self.cost.types(), self.cost.all_tokens()))
            _logger.info("Unweighted corpus cost: %s lexicon cost: %s" % (cc, lc))
            if done:
                _logger.info('Reached pruning goal')
        return epoch, cost

    def train_batch(self, algorithm='recursive', algorithm_params=(),
                    finish_threshold=0.005, max_epochs=None):
        """Train the model in batch fashion.

        The model is trained with the data already loaded into the model (by
        using an existing model or calling one of the load\_ methods).

        In each iteration (epoch) all compounds in the training data are
        optimized once, in a random order. If applicable, corpus weight,
        annotation cost, and random split counters are recalculated after
        each iteration.

        Arguments:
            algorithm: string in ('recursive', 'viterbi', 'flatten') 
                         that indicates the splitting algorithm used.
            algorithm_params: parameters passed to the splitting algorithm.
            finish_threshold: the stopping threshold. Training stops when
                                the improvement of the last iteration is
                                smaller then finish_threshold * #boundaries
            max_epochs: maximum number of epochs to train

        """
        self._check_normal_mode()
        epochs = 0
        forced_epochs = max(1, self._epoch_update(epochs))
        newcost = self.get_cost()
        compounds = list(self.get_compounds())
        _logger.info("Compounds in training data: %s types / %s tokens" %
                     (len(compounds), self.cost.compound_tokens()))

        if algorithm == 'flatten':
            _logger.info("Flattening analysis tree")
            for compound in _progress(compounds):
                parts = self.segment(compound)
                self._clear_compound_analysis(compound)
                self._set_compound_analysis(compound, parts)
            _logger.info("Done.")
            return 1, self.get_cost()

        _logger.info("Starting batch training")
        _logger.info("Epochs: %s\tCost: %s" % (epochs, newcost))

        while True:
            # One epoch
            random.shuffle(compounds)

            for w in _progress(compounds):
                if algorithm == 'recursive':
                    segments = self._recursive_optimize(w, *algorithm_params)
                elif algorithm == 'viterbi':
                    segments = self._viterbi_optimize(w, *algorithm_params)
                else:
                    raise MorfessorException("unknown algorithm '%s'" %
                                             algorithm)
                _logger.debug("#%s -> %s" %
                              (w, " + ".join(self.cc.to_string(s) for s in segments)))
            epochs += 1

            _logger.debug("Cost before epoch update: %s" % self.get_cost())
            forced_epochs = max(forced_epochs, self._epoch_update(epochs))
            oldcost = newcost
            newcost = self.get_cost()
            lc, cc = self.cost.cost_before_tuning()

            self._epoch_checks()

            _logger.info("Epochs: %s\tCost: %s" % (epochs, newcost))
            _logger.info("Unweighted corpus cost: %s lexicon cost: %s" % (cc, lc))
            if (forced_epochs == 0 and
                    newcost >= oldcost - finish_threshold *
                    self.cost.compound_tokens()):
                break
            if forced_epochs > 0:
                forced_epochs -= 1
            if max_epochs is not None and epochs >= max_epochs:
                _logger.info("Max number of epochs reached, stop training")
                break
        _logger.info("Done.")
        return epochs, newcost

    def train_online(self, data, count_modifier=None, epoch_interval=10000,
                     algorithm='recursive', algorithm_params=(),
                     init_rand_split=None, max_epochs=None):
        """Train the model in online fashion.

        The model is trained with the data provided in the data argument.
        As example the data could come from a generator linked to standard in
        for live monitoring of the splitting.

        All compounds from data are only optimized once. After online
        training, batch training could be used for further optimization.

        Epochs are defined as a fixed number of compounds. After each epoch (
        like in batch training), the annotation cost, and random split counters
        are recalculated if applicable.

        Arguments:
            data: iterator of (_, compound_atoms) tuples. The first
                    argument is ignored, as every occurence of the
                    compound is taken with count 1
            count_modifier: function for adjusting the counts of each
                              compound
            epoch_interval: number of compounds to process before starting
                              a new epoch
            algorithm: string in ('recursive', 'viterbi') that indicates
                         the splitting algorithm used.
            algorithm_params: parameters passed to the splitting algorithm.
            init_rand_split: probability for random splitting a compound to
                               at any point for initializing the model. None
                               or 0 means no random splitting.
            max_epochs: maximum number of epochs to train

        """
        self._check_normal_mode()
        if count_modifier is not None:
            counts = {}

        _logger.info("Starting online training")

        epochs = 0
        i = 0
        more_tokens = True
        while more_tokens:
            self._epoch_update(epochs)
            newcost = self.get_cost()
            _logger.info("Tokens processed: %s\tCost: %s" % (i, newcost))

            for _ in _progress(range(epoch_interval)):
                try:
                    dp = next(data)
                except StopIteration:
                    more_tokens = False
                    break

                self._add_compound(dp.compound, dp.count)
                self._clear_compound_analysis(dp.compound)
                self._set_compound_analysis(dp.compound, self.cc.splitn(dp.compound, dp.splitlocs))

                if algorithm == 'recursive':
                    segments = self._recursive_optimize(dp.compound, *algorithm_params)
                elif algorithm == 'viterbi':
                    segments = self._viterbi_optimize(dp.compound, *algorithm_params)
                else:
                    raise MorfessorException("unknown algorithm '%s'" %
                                             algorithm)
                _logger.debug("#%s: %s -> %s" %
                              (i, dp.compound, " + ".join(self.cc.to_string(s) for s in segments)))
                i += 1

            epochs += 1
            if max_epochs is not None and epochs >= max_epochs:
                _logger.info("Max number of epochs reached, stop training")
                break

        self._epoch_update(epochs)
        newcost = self.get_cost()
        _logger.info("Tokens processed: %s\tCost: %s" % (i, newcost))
        return epochs, newcost

    def viterbi_segment(self, compound, addcount=1.0, maxlen=30,
                        allow_longer_unk_splits=False,
                        taboo=None):
        """Find optimal segmentation using the Viterbi algorithm.

        Arguments:
          compound: compound to be segmented
          addcount: constant for additive smoothing (0 = no smoothing)
          maxlen: maximum length for the constructions
          taboo: not allowed to use these constructions

        If additive smoothing is applied, new complex construction types can
        be selected during the search. Without smoothing, only new
        single-atom constructions can be selected.

        Returns the most probable segmentation and its log-probability.

        """
        #clen = len(compound)
        # indices = range(1, clen+1) if allowed_boundaries is None \
        #           else allowed_boundaries+[clen]

        grid = {None: (0.0, None)}
        tokens = self.cost.all_tokens() + addcount
        logtokens = math.log(tokens) if tokens > 0 else 0
        taboo = set() if taboo is None else set(taboo)

        newboundcost = self.cost.newbound_cost(addcount) if addcount > 0 else 0

        badlikelihood = self.cost.bad_likelihood(compound,addcount)

        for t in itertools.chain(self.cc.split_locations(compound), [None]):
            # Select the best path to current node.
            # Note that we can come from any node in history.
            bestpath = None
            bestcost = None

            for pt in tail(maxlen, itertools.chain([None], self.cc.split_locations(compound, stop=t))):
                if grid[pt][0] is None:
                    continue
                cost = grid[pt][0]
                construction = self.cc.slice(compound, pt, t)
                if construction in taboo:
                    continue
                count = self.get_construction_count(construction)
                if count > 0:
                    cost += (logtokens - math.log(count + addcount))
                elif addcount > 0:
                    if self.cost.tokens() == 0:
                        cost += (addcount * math.log(addcount) +
                                newboundcost + self.cost.get_coding_cost(construction))
                    else:
                        cost += (logtokens - math.log(addcount) +
                                newboundcost + self.cost.get_coding_cost(construction))

                elif self.cc.is_atom(construction):
                    cost += badlikelihood
                elif allow_longer_unk_splits:
                    # Some splits are forbidden, so longer unknown
                    # constructions have to be allowed
                    cost += len(self.cc.corpus_key(construction)) * badlikelihood
                else:
                    continue
                #_logger.debug("cost(%s)=%.2f", construction, cost)
                if bestcost is None or cost < bestcost:
                    bestcost = cost
                    bestpath = pt
            grid[t] = (bestcost, bestpath)

        splitlocs = []

        cost, path = grid[None]
        while path is not None:
            splitlocs.append(path)
            path = grid[path][1]

        constructions = list(self.cc.splitn(compound, list(reversed(splitlocs))))

        # Add boundary cost
        if self._mode == MODE_NORMAL:
            cost += (math.log(self.cost.tokens() +
                            self.cost.compound_tokens()) -
                    math.log(self.cost.compound_tokens()))
        return constructions, cost

    def sample_segment(self, compound, theta=0.5, maxlen=30):
        """Sample a segmentation using the
        Forward-filter Backward-sample algorithm.

        Arguments:
          compound: compound to be segmented
          theta: sampling temperature. (1.0 = Viterbi).
          maxlen: maximum length for the constructions

        Returns the sampled segmentation and its log-probability.

        """
        grid = {'start': (0.0, None)}
        tokens = self.cost.all_tokens()
        logtokens = math.log(tokens) if tokens > 0 else 0

        badlikelihood = self.cost.bad_likelihood(compound, 0)
        extrabad = badlikelihood**2

        if len(compound) == 1:
            return [compound], 0

        ## Forward filtering pass
        for t in itertools.chain(self.cc.split_locations(compound), ['stop']):
            # logsum of all paths to current node.
            # Note that we can come from any node in history.
            negcosts = []

            for pt in tail(maxlen, itertools.chain(['start'], self.cc.split_locations(compound, stop=t))):
                if grid[pt][0] is None:
                    continue
                cost = grid[pt][0]
                construction = self.cc.slice(compound, pt, t)
                count = self.get_construction_count(construction)
                if count > 0:
                    cost += (logtokens - theta * math.log(count))
                elif self.cc.is_atom(construction):
                    cost += badlikelihood
                else:
                    continue
                #_logger.debug("cost(%s)=%.2f", construction, cost)
                negcosts.append(-cost)
            if len(negcosts) == 0:
                grid[t] = (extrabad, None)
                continue
            totcost = -logsumexp(negcosts)
            grid[t] = (totcost, None)

        ## Backward sampling pass
        splitlocs = []
        t = 'stop'
        totcost = grid['stop'][0]
        while t is not None:
            pts = []
            probs = []
            t = None if t == 'stop' else t
            for pt in tail(maxlen, itertools.chain(['start'], self.cc.split_locations(compound, stop=t))):
                if grid[pt][0] is None:
                    continue
                cost = grid[pt][0]
                # cc.slice requires None for endpoints
                pt = None if pt == 'start' else pt
                construction = self.cc.slice(compound, pt, t)
                count = self.get_construction_count(construction)
                if count > 0:
                    cost += theta * (logtokens - math.log(count) - totcost)
                elif self.cc.is_atom(construction):
                    cost += badlikelihood
                else:
                    continue
                pts.append(pt)
                if cost < 0:
                    # FIXME: bug or imprecision?
                    probs.append(1)
                else:
                    probs.append(math.exp(-cost))
            if sum(probs) < EPS:
                # if noting is valid, letterize
                if t is None:
                    t = len(compound)
                sample = t - 1
                if sample <= 0:
                    sample = None
            else:
                sample = categorical(pts, probs)
            if sample is not None:
                splitlocs.append(sample)
            t = sample

        constructions = list(self.cc.splitn(compound, list(reversed(splitlocs))))

        return constructions, cost

    def forward_backward(self, compound, freq, maxlen=30):
        grid_alpha = {'start': (0.0, None)}
        grid_beta = {'stop': (0.0, None)}
        tokens = self.cost.tokens()
        logtokens = math.log(tokens) if tokens > 0 else 0

        local_morph_costs = {}

        badlikelihood = self.cost.bad_likelihood(compound, 0)

        ## Forward pass
        for t in itertools.chain(self.cc.split_locations(compound), ['stop']):
            # logsum of all paths to current node.
            # Note that we can come from any node in history.
            negcosts = []

            for pt in tail(maxlen, itertools.chain(['start'], self.cc.split_locations(compound, stop=t))):
                if grid_alpha[pt][0] is None:
                    continue
                construction = self.cc.slice(compound, pt, t)
                if construction not in local_morph_costs:
                    count = self.get_construction_count(construction)
                    if count > 0:
                        cost = (logtokens - math.log(count))
                    elif self.cc.is_atom(construction):
                        cost = badlikelihood
                    else:
                        local_morph_costs[construction] = None
                        continue
                    assert cost >= 0
                    local_morph_costs[construction] = cost
                cost = local_morph_costs[construction]
                if cost is None:
                    continue
                cost += grid_alpha[pt][0]
                #_logger.debug("cost(%s)=%.2f", construction, cost)
                negcosts.append(-cost)
            totcost = -logsumexp(negcosts)
            grid_alpha[t] = (totcost, None)

        ## Backward pass
        for t in itertools.chain(reversed(list(self.cc.split_locations(compound))), ['start']):
            negcosts = []
            for pt in itertools.islice(
                    itertools.chain(self.cc.split_locations(compound, start=t), ['stop']), maxlen):
                if grid_beta[pt][0] is None:
                    continue
                construction = self.cc.slice(compound, t, pt)
                cost = local_morph_costs[construction]
                if cost is None:
                    continue
                cost += grid_beta[pt][0]
                negcosts.append(-cost)
            totcost = -logsumexp(negcosts)
            grid_beta[t] = (totcost, None)

        ## Merge pass
        w_expected = collections.Counter()
        totcost = grid_alpha['stop'][0]
        # grid_alpha['stop'][0], grid_beta['start'][0] are approx equal
        for t in itertools.chain(self.cc.split_locations(compound), ['stop']):
            for pt in tail(maxlen, itertools.chain(['start'], self.cc.split_locations(compound, stop=t))):
                # grid_alpha[pt][0] is the total probability of all paths ending at pt
                # grid_beta[t][0] is the total probability of all paths starting at t
                # the compound pt:t probability is the same as cached previously
                if grid_alpha[pt][0] is None:
                    continue
                if grid_beta[t][0] is None:
                    continue
                construction = self.cc.slice(compound, pt, t)
                cost = local_morph_costs[construction]
                if cost is None:
                    continue
                expect = math.exp(
                    -grid_alpha[pt][0] -grid_beta[t][0] -cost + totcost)
                if expect > 1:
                    occurs = compound.count(construction)
                    assert expect <= occurs + EPS, '"{}" has expect {} occurs {}'.format(
                        construction, expect, occurs)
                w_expected[construction] += freq * expect

        return w_expected, freq * totcost

    #TODO project lambda
    def forward_logprob(self, compound):
        """Find log-probability of a compound using the forward algorithm.

        Arguments:
          compound: compound to process

        Returns the (negative) log-probability of the compound. If the
        probability is zero, returns a number that is larger than the
        value defined by the penalty attribute of the model object.

        """
        clen = len(compound)
        grid = [0.0]
        if self._corpus_coding.tokens + self._corpus_coding.boundaries > 0:
            logtokens = math.log(self._corpus_coding.tokens +
                                 self._corpus_coding.boundaries)
        else:
            logtokens = 0
        # Forward main loop
        for t in range(1, clen + 1):
            # Sum probabilities from all paths to the current node.
            # Note that we can come from any node in history.
            psum = 0.0
            for pt in range(0, t):
                cost = grid[pt]
                construction = compound[pt:t]
                count = self.get_construction_count(construction)
                if count > 0:
                    cost += (logtokens - math.log(count))
                else:
                    continue
                psum += math.exp(-cost)
            if psum > 0:
                grid.append(-math.log(psum))
            else:
                grid.append(-self.penalty)
        cost = grid[-1]
        # Add boundary cost
        cost += (math.log(self._corpus_coding.tokens +
                          self._corpus_coding.boundaries) -
                 math.log(self._corpus_coding.boundaries))
        return cost

    def viterbi_nbest(self, compound, n, addcount=1.0, theta=1.0, maxlen=30,
                      allow_longer_unk_splits=False):
        """Find top-n optimal segmentations using the Viterbi algorithm.

        Arguments:
          compound: compound to be segmented
          n: how many segmentations to return
          addcount: constant for additive smoothing (0 = no smoothing)
          theta: sampling temperature. (1.0 = Viterbi).
          maxlen: maximum length for the constructions

        If additive smoothing is applied, new complex construction types can
        be selected during the search. Without smoothing, only new
        single-atom constructions can be selected.

        Returns the n most probable segmentations and their
        log-probabilities.

        """
        grid = {None: [(0.0, None)]}
        tokens = self.cost.all_tokens() + addcount
        logtokens = math.log(tokens) if tokens > 0 else 0

        newboundcost = self.cost.newbound_cost(addcount) if addcount > 0 else 0

        badlikelihood = self.cost.bad_likelihood(compound,addcount)

        # Viterbi main loop
        for t in itertools.chain(self.cc.split_locations(compound), [None]):
            # Select the best path to current node.
            # Note that we can come from any node in history.
            bestn = []
            for pt in tail(maxlen, itertools.chain([None], self.cc.split_locations(compound, stop=t))):
                for k in range(len(grid[pt])):
                    if grid[pt][k][0] is None:
                        continue
                    cost = -grid[pt][k][0]
                    construction = self.cc.slice(compound, pt, t)
                    count = self.get_construction_count(construction)
                    if count > 0:
                        cost += (logtokens - theta * math.log(count + addcount))
                    elif addcount > 0:
                        if self.cost.tokens() == 0:
                            cost += (addcount * math.log(addcount) +
                                    newboundcost + self.cost.get_coding_cost(construction))
                        else:
                            cost += (logtokens - math.log(addcount) +
                                    newboundcost + self.cost.get_coding_cost(construction))

                    elif self.cc.is_atom(construction):
                        cost += badlikelihood
                    elif allow_longer_unk_splits:
                        # Some splits are forbidden, so longer unknown
                        # constructions have to be allowed
                        cost += len(self.cc.corpus_key(construction)) * badlikelihood
                    else:
                        continue
                    if len(bestn) < n:
                        heapq.heappush(bestn, (-cost, pt, k))
                    else:
                        heapq.heappushpop(bestn, (-cost, pt, k))
            grid[t] = bestn
        results = []
        for k in range(len(grid[None])):
            constructions = []
            cost, path, ki = grid[None][k]
            cost = -cost
            lt = None
            if path is None:
                constructions = [compound]
            else:
                while True:
                    t = path
                    constructions.append(self.cc.slice(compound, t, lt))
                    path = grid[t][ki][1]
                    ki = grid[t][ki][2]
                    lt = t
                    if lt is None:
                        break
            constructions.reverse()
            # Add boundary cost
            if self._mode == MODE_NORMAL:
                cost += (math.log(self.cost.tokens() +
                                self.cost.compound_tokens()) -
                        math.log(self.cost.compound_tokens()))
            results.append((cost, constructions))
        if len(results) == 0:
            results = [(badlikelihood, [compound])]
        return [(constr, cost) for cost, constr in sorted(results)]

    def get_corpus_coding_weight(self):
        return self.cost._corpus_coding.weight

    def set_corpus_coding_weight(self, weight):
        self._check_segment_only()
        self.cost.set_corpus_coding_weight(weight)

    def make_segment_only(self):
        """Reduce the size of this model by removing all non-morphs from the
        analyses. After calling this method it is not possible anymore to call
        any other method that would change the state of the model. Anyway
        doing so would throw an exception.

        """
        self._check_normal_mode()
        self._num_compounds = len(self.get_compounds())
        self._segment_only = True

        self._analyses = {k: v for (k, v) in self._analyses.items()
                          if not v.splitloc}

    def clear_segmentation(self):
        self._check_normal_mode()
        for compound in self.get_compounds():
            self._clear_compound_analysis(compound)
            self._set_compound_analysis(compound, [compound])

    def get_params(self):
        """Returns a dict of hyperparameters."""
        params = {'corpusweight': self.get_corpus_coding_weight()}
        #if self._supervised:
        #    params['annotationweight'] = self._annot_coding.weight
        params['forcesplit'] = ''.join(sorted(self.cc._force_splits))
        if self.cc._nosplit:
            params['nosplit'] = self.cc._nosplit.pattern
        return params

# count = count of the node
# splitloc = integer or tuple. Location(s) of the possible splits for virtual
#            constructions; empty tuple or 0 if real construction
SimpleConstrNode = collections.namedtuple('ConstrNode', ['count', 'splitloc'])
