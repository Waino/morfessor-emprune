import collections


class ParallelConstructionMethods(object):
    type = collections.namedtuple("ParallelConstruction", ['graphemes', 'phonemes'])

    @staticmethod
    def force_split_locations(construction):
        return []

    @staticmethod
    def split_locations(construction, start=None, stop=None):
        """
        Return all possible split-locations between start and end. Start and end will not be returned.
        """
        start = (0,0) if start is None else start
        end = (len(construction.graphemes), len(construction.phonemes)) if stop is None else stop

        for gi in range(start[0] + 1, end[0]):
            for pi in range(start[1] + 1, end[1]):
                yield (gi, pi)

    @classmethod
    def split(cls, construction, loc):
        assert 0 < loc[0] < len(construction.graphemes)
        assert 0 < loc[1] < len(construction.phonemes)
        return (cls.type(construction.graphemes[:loc[0]], construction.phonemes[:loc[1]]),
               cls.type(construction.graphemes[loc[0]:], construction.phonemes[loc[1]:]))

    @classmethod
    def splitn(cls, construction, locs):
        if len(locs) > 0 and not hasattr(locs[0], '__iter__'):
            for p in cls.split(construction, locs):
                yield p
            return

        prev = (0,0)
        for l in locs:
            assert prev[0] < l[0] < len(construction.graphemes)
            assert prev[1] < l[1] < len(construction.phonemes)
            yield cls.type(construction.graphemes[prev[0]:l[0]], construction.phonemes[prev[1]:l[1]])
            prev = l
        yield cls.type(construction.graphemes[prev[0]:], construction.phonemes[prev[1]:])

    @staticmethod
    def parts_to_splitlocs(parts):
        cur_len = [0, 0]
        for p in parts[:-1]:
            cur_len[0] += len(p.graphemes)
            cur_len[1] += len(p.phonemes)
            yield tuple(cur_len)

    @classmethod
    def slice(cls, construction, start=None, stop=None):
        start = (0,0) if start is None else start
        stop = (len(construction.graphemes), len(construction.phonemes)) if stop is None else stop
        return cls.type(construction.graphemes[start[0]:stop[0]],
                        construction.phonemes[start[1]:stop[1]])

    @classmethod
    def from_string(cls, string):
        g, p = string.split('/', 1)
        assert len(g) > 0
        assert len(p) > 0
        return cls.type(g, p)

    @staticmethod
    def to_string(construction):
        return u"{}/{}".format(construction.graphemes, construction.phonemes)

    @staticmethod
    def corpus_key(construction):
        return construction

    @staticmethod
    def lex_key(construction):
        a = []
        a.extend(construction.graphemes)
        a.extend(construction.phonemes)
        return tuple(a)

    @staticmethod
    def atoms(construction):
        a = []
        a.extend(construction.graphemes)
        a.extend(construction.phonemes)
        return tuple(a)