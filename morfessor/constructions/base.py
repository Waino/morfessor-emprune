import re


class BaseConstructionMethods(object):
    def __init__(self, force_splits=None, nosplit_re=None):
        self._force_splits = set(force_splits) if force_splits is not None else set()
        self._nosplit = re.compile(nosplit_re, re.UNICODE) if nosplit_re is not None else None

    def force_split_locations(self, construction):
        prev = 0
        for i in range(len(construction)):
            if construction[i] in self._force_splits:
                if i-prev > 0:
                    yield i
                if i+1 < len(construction):
                    yield i+1
                prev = i+1

    def split_locations(self, construction, start=None, stop=None):
        """
        Return all possible split-locations between start and end. Start and end will not be returned.
        """
        start = start if start is not None else 0
        stop = stop if stop is not None else len(construction)

        for i in range(start+1, stop):
            if self._nosplit and self._nosplit.match(construction[i-1:i+1]):
                continue
            yield i

    @staticmethod
    def split(construction, loc):
        assert 0 < loc < len(construction)
        return construction[:loc], construction[loc:]

    @classmethod
    def splitn(cls, construction, locs):
        if not hasattr(locs, '__iter__'):
            for p in cls.split(construction, locs):
                yield p
            return

        prev = 0
        for l in locs:
            assert prev < l < len(construction)
            yield construction[prev:l]
            prev = l
        yield construction[prev:]

    @staticmethod
    def parts_to_splitlocs(parts):
        cur_len = 0
        for p in parts[:-1]:
            cur_len += len(p)
            yield cur_len

    @staticmethod
    def slice(construction, start=None, stop=None):
        return construction[start:stop]

    @staticmethod
    def from_string(string):
        return string

    @staticmethod
    def to_string(construction):
        return construction

    @staticmethod
    def corpus_key(construction):
        return construction

    @staticmethod
    def lex_key(construction):
        return construction

    @staticmethod
    def atoms(construction):
        return construction

    @classmethod
    def is_atom(cls, construction):
        return len(cls.corpus_key(construction)) == 1
