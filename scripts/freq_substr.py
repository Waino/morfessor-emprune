#!/usr/bin/env python3
import collections
import itertools
import sys
from morfessor.utils import SortedCounter

def substrs(seq, min_len=1, max_len=None, prefixes=None):
    prefix_len = min_len - 1
    len_seq = len(seq)
    max_len = len_seq if max_len is None else max_len
    n_lens = max_len - prefix_len
    for start in range(len_seq - prefix_len):
        if prefixes is not None:
            prefix = seq[start:(start + prefix_len)]
            if prefix not in prefixes:
                continue
        for j in range(min(n_lens, len_seq - start - prefix_len)):
            end = start + j + prefix_len + 1
            yield seq[start:end]

class FrequentSubstrings(object):
    def __init__(self,
                 lex_size,
                 prefix_len,
                 prune_redundant,
                 forcesplit_before,
                 forcesplit_after):
        self.lex_size = lex_size
        self.prefix_len = prefix_len
        self.prune_redundant_margin = prune_redundant
        self.forcesplit_before = set(forcesplit_before)
        self.forcesplit_after = set(forcesplit_after)

        self.do_forcesplit = (len(self.forcesplit_before) + len(self.forcesplit_after)) > 0
        self.chars = collections.Counter()
        self.prefixes = SortedCounter()
        self.long = collections.Counter()

    def first_pass(self, seqs):
        # compute frequency of short substrings
        for count, seq in seqs:
            for char in seq:
                self.chars[char] += count
            for substr in substrs(seq, max_len=self.prefix_len):
                self.prefixes[substr] += count
        # prune out infrequent substrings
        if len(self.prefixes) > self.lex_size:
            self.prefixes = collections.Counter(
                dict(self.prefixes.most_common(self.lex_size)))
        else:
            print('Nothing pruned in first pass. Use longer prefixes.', file=sys.stderr)

    def second_pass(self, seqs):
        # extend frequent prefixes
        for count, seq in seqs:
            for substr in substrs(seq, min_len=self.prefix_len + 1, prefixes=self.prefixes):
                self.long[substr] += count

    def _is_redundant(self, count, prefix_count):
        return prefix_count <= (count + self.prune_redundant_margin)

    def prune_redundant(self, seed_lexicon):
        pruned = set()
        for subw, count in seed_lexicon.items():
            # test all prefixes, break early
            for i in range(1, len(subw)):
                prefix = subw[:-i]
                prefix_count = seed_lexicon[prefix]
                if self._is_redundant(count, prefix_count):
                    pruned.add(prefix)
                else:
                    break
            # test all suffixes, break early
            for i in range(1, len(subw)):
                suffix = subw[i:]
                suffix_count = seed_lexicon[suffix]
                if self._is_redundant(count, suffix_count):
                    pruned.add(suffix)
                else:
                    break
        for subw in pruned:
            del seed_lexicon[subw]

    def forcesplit(self, seed_lexicon):
        pruned = set()
        for subw in seed_lexicon:
            if len(subw) == 1:
                # always keep chars
                continue
            for char in subw[1:]:
                # non-initial position
                if char in self.forcesplit_before:
                    pruned.add(subw)
                    break
            for char in subw[:-1]:
                # non-final position
                if char in self.forcesplit_after:
                    pruned.add(subw)
                    break
        for subw in pruned:
            del seed_lexicon[subw]

    def finalize(self):
        n_rare_chars = sum(1 for c in self.chars if c not in self.prefixes)
        prune_to = self.lex_size - n_rare_chars
        combined = self.prefixes
        combined.update(self.long)
        if self.do_forcesplit:
            self.forcesplit(combined)
        if self.prune_redundant_margin >= 0:
            self.prune_redundant(combined)
        # prune out infrequent substrings
        if len(combined) > prune_to:
            combined = SortedCounter(
                dict(combined.most_common(prune_to)))
        else:
            print('Nothing pruned in finalize. Your total lexicon is small.', file=sys.stderr)
        for char, count in self.chars.items():
            if char not in combined:
                combined[char] = count
        del self.chars
        del self.prefixes
        del self.long
        return SortedCounter(combined)

def get_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="""
Morfessor EM+Prune %s

Copyright (c) 2019, Stig-Arne Grönroos, Sami Virpioja, and Peter Smit.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1.  Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

2.  Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided
    with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Command-line arguments:
        """,
        epilog="""
Simple usage:
    freq_substr.py < input > output
        """)
    parser.add_argument('--lex-size', type=int, default=1000000,
        help='Size of seed lexicon')
    parser.add_argument('--prefix', type=int, default=6,
        help='Length of prefix to use in first pruning pass')
    parser.add_argument('--prune-redundant', type=int, default=0,
        help='Prune redundant prefix and suffix substrings. '
             'Set to -1 to disable the pruning. '
             'Set to 0 to prune only when counts are equal. '
             'Set to a positive number to prune more aggressively. ')
    parser.add_argument('--forcesplit-before', type=str, default='',
        help='Characters to force splitting before. '
             'Substrings with these chars in non-initial position will be pruned out.')
    parser.add_argument('--forcesplit-after', type=str, default='',
        help='Characters to force splitting after. '
             'Substrings with these chars in non-final position will be pruned out.')
    parser.add_argument('--forcesplit-both', type=str, default='',
        help='Characters to force splitting both before and after. '
             'Substrings longer than 1 with these chars will be pruned out.')
    parser.add_argument('--traindata-list', dest="list", default=False,
            action='store_true',
            help="input file is a word-count list "
                 "(one compound per line, preceded by count of compound)."
                 "Recommended.")

    return parser


def read_word_counts(lines):
    lines = [line.strip().split(None, 1) for line in lines]
    counts = [(int(count), word) for (count, word) in lines]
    return counts


def read_corpus(lines):
    counts = collections.Counter()
    for line in lines:
        counts.update(line.split())
    counts = [(count, word) for (word, count) in counts.most_common()]
    return counts


def main():
    parser = get_parser()
    args = parser.parse_args()
    lines = sys.stdin
    if args.list:
        lines = read_word_counts(lines)
    else:
        lines = read_corpus(lines)
    forcesplit_before = args.forcesplit_before + args.forcesplit_both
    forcesplit_after = args.forcesplit_after + args.forcesplit_both
    freq_substrs = FrequentSubstrings(
        args.lex_size,
        args.prefix,
        args.prune_redundant,
        forcesplit_before,
        forcesplit_after)
    freq_substrs.first_pass(lines)
    freq_substrs.second_pass(lines)
    combined = freq_substrs.finalize()
    for w, c in combined.most_common():
        print('{} {}'.format(c, w))

if __name__ == '__main__':
    main()
