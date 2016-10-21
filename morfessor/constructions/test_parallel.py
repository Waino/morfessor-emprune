import unittest

from morfessor.constructions.parallel import ParallelConstructionMethods


class TestParallelConstruction(unittest.TestCase):
    def setUp(self):
        self.cc = ParallelConstructionMethods()

    def test_force_split_locations(self):
        # ParallelConstruction does not support force_splits, so always a iterable of length 0 should be returned
        for constr in (None, self.cc.type(u"hi", u"hello")):
            self.assertEqual(len(list(self.cc.force_split_locations(constr))), 0,
                             u"force_split_lcoation should return 0-length iterable")

    def test_split_locations(self):
        constr = self.cc.type(u"hi!", u"hello")
        self.assertListEqual(list(self.cc.split_locations(constr)),
                             [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4)])

        self.assertListEqual(list(self.cc.split_locations(constr, start=(1, 2))), [(2, 3), (2, 4)])
        self.assertListEqual(list(self.cc.split_locations(constr, start=(2, 1))), [])

        self.assertListEqual(list(self.cc.split_locations(constr, stop=(2, 2))), [(1, 1)])
        self.assertListEqual(list(self.cc.split_locations(constr, stop=(1, 1))), [])

    def test_split(self):
        constr = self.cc.type(u"hi!", u"hello")

        for false_split in [(0,0), (1, 0), (0, 1), (3,5), (3,0), (0,5)]:
            self.assertRaises(AssertionError, self.cc.split, constr, false_split)

        self.assertEqual(tuple(self.cc.split(constr, (1, 2))), (self.cc.type(u"h", u"he"), self.cc.type(u"i!", u"llo")))

    def test_splitn(self):
        constr = self.cc.type(u"hi!", u"hello")

        #false single splits
        for false_split in [(0,0), (1, 0), (0, 1), (3,5), (3,0), (0,5)]:
            with self.assertRaises(AssertionError, msg="splitn with false_split {}".format(false_split)):
                list(self.cc.splitn(constr, false_split))

        #false iterable splits
        for false_split in [(0,0), (1, 0), (0, 1), (3,5), (3,0), (0,5)]:
            with self.assertRaises(AssertionError, msg="splitn with false_split [{}]".format(false_split)):
                list(self.cc.splitn(constr, [false_split]))

        #empty split iterable
        self.assertListEqual(list(self.cc.splitn(constr, [])), [constr])

        #ok single split
        self.assertListEqual(list(self.cc.splitn(constr, (1,2))), [self.cc.type(u"h", u"he"), self.cc.type(u"i!", u"llo")])
        self.assertListEqual(list(self.cc.splitn(constr, [(2, 4)])), [self.cc.type(u"hi", u"hell"), self.cc.type(u"!", u"o")])

        #ok double split
        self.assertListEqual(list(self.cc.splitn(constr, [(1,2),(2, 4)])), [self.cc.type(u"h", u"he"), self.cc.type(u"i", u"ll"), self.cc.type(u"!", u"o")])

    def test_parts_to_splitloc(self):
        constr = self.cc.type(u"hi!", u"hello")
        self.assertListEqual(list(self.cc.parts_to_splitlocs([])), [])
        self.assertListEqual(list(self.cc.parts_to_splitlocs([constr])), [])
        self.assertListEqual(list(self.cc.parts_to_splitlocs([constr, constr])), [(3,5)])
        self.assertListEqual(list(self.cc.parts_to_splitlocs([constr, constr, constr])), [(3, 5), (6,10)])

    def test_slice(self):
        constr = self.cc.type(u"hi!", u"hello")

        self.assertEqual(self.cc.slice(constr), constr)
        self.assertEqual(self.cc.slice(constr, start=(0, 0)), constr)
        self.assertEqual(self.cc.slice(constr, stop=(3, 5)), constr)

        self.assertEqual(self.cc.slice(constr, start=(1, 0)), self.cc.type(u"i!", u"hello"))
        self.assertEqual(self.cc.slice(constr, start=(0, 2)), self.cc.type(u"hi!", u"llo"))
        self.assertEqual(self.cc.slice(constr, start=(1, 2)), self.cc.type(u"i!", u"llo"))

    def test_from_string(self):
        self.assertEqual(self.cc.from_string(u"hi!/hello"), self.cc.type(u"hi!", u"hello"))
        self.assertEqual(self.cc.from_string(u"hi!/hello//"), self.cc.type(u"hi!", u"hello//"))

        for fail_string in (u"hi!/", u"/hello", u"/"):
            with self.assertRaises(AssertionError):
                _ = self.cc.from_string(fail_string)

        with self.assertRaises(ValueError):
            _ = self.cc.from_string(u"hi")

    def test_to_string(self):
        self.assertEqual(self.cc.to_string(self.cc.type(u"hi!", u"hello")), u"hi!/hello")

    def test_corpus_key(self):
        constr = self.cc.type(u"hi!", u"hello")
        self.assertEqual(self.cc.corpus_key(constr), constr)

    def test_lex_key(self):
        constr = self.cc.type(u"hi!", u"hello")
        self.assertEqual(self.cc.lex_key(constr), constr.graphemes)

    def test_atoms(self):
        constr = self.cc.type(u"hi!", u"hello")
        self.assertEqual(self.cc.atoms(constr), constr.graphemes)