from collections import Counter, namedtuple
from random import random


DataPoint = namedtuple('DataPoint', ['count', 'compound', 'splitlocs'])


def merge_counts(data):
    store = {}
    for dp in data:
        if dp.compound in store:
            store[dp.compound] = dp._replace(count=store[dp.compound].count + dp.count)
        else:
            store[dp.compound] = dp

    for v in sorted(store.values()):
        yield v

def freq_threshold(data, threshold, online=False):
    if online:
        counts = Counter()
        for dp in data:
            yielded = counts[dp.compound] if counts[dp.compound] >= threshold else 0
            counts[dp.compound] += dp.count

            if counts[dp.compound] >= threshold:
                yield dp._replace(count=dp.count-yielded)

    else:
        for dp in data:
            if dp.count >= threshold:
                yield dp


def count_modifier(data, modifier, online=False):
    if online:
        counts = Counter()
        for dp in data:
            old_val = 0
            if dp.compound in counts:
                old_val = modifier(counts[dp.compound])
            counts[dp.compound] += dp.count
            new_val = modifier(counts[dp.compound])

            if new_val - old_val > 0:
                yield dp._replace(count=new_val-old_val)

    else:
        for dp in data:
            yield dp._replace(count=modifier(dp.count))


def rand_split(data, cc, threshold, rand_gen=random):
    for dp in data:
        forced = cc.force_split_locations(dp.compound)
        all = cc.split_locations(dp.compound)
        yield dp._replace(splitlocs=tuple(i for i in all if (i in forced or rand_gen() < threshold)))