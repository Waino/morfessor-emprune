#!/usr/bin/env python3
import collections
import itertools
import sys

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
    def __init__(self, lex_size, prefix_len):
        self.lex_size = lex_size
        self.prefix_len = prefix_len
        self.chars = collections.Counter()
        self.prefixes = collections.Counter()
        self.long = collections.Counter()

    def first_pass(self, seqs):
        # compute frequency of short substrings
        for count, seq in seqs:
            self.chars.update(seq)
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

    def finalize(self):
        n_rare_chars = sum(1 for c in self.chars if c not in self.prefixes)
        prune_to = self.lex_size - n_rare_chars
        combined = self.prefixes
        combined.update(self.long)
        # prune out infrequent substrings
        if len(combined) > prune_to:
            combined = collections.Counter(
                dict(combined.most_common(prune_to)))
        else:
            print('Nothing pruned in finalize. Your total lexicon is small.', file=sys.stderr)
        for char, count in self.chars.items():
            if char not in combined:
                combined[char] = count
        del self.chars
        del self.prefixes
        del self.long
        return combined

def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lex-size', type=int, default=25000)
    parser.add_argument('--prefix', type=int, default=4)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    lines = sys.stdin
    lines = [line.strip().split(None, 1) for line in lines]
    lines = [(int(count), word) for (count, word) in lines]
    freq_substrs = FrequentSubstrings(args.lex_size, args.prefix)
    freq_substrs.first_pass(lines)
    freq_substrs.second_pass(lines)
    combined = freq_substrs.finalize()
    for w, c in combined.most_common():
        print('{} {}'.format(c, w))

if __name__ == '__main__':
    main()
