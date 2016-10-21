import unittest

from morfessor.constructions.base import BaseConstructionMethods


class TestParallelConstruction(unittest.TestCase):
    def setUp(self):
        self.cc = BaseConstructionMethods(force_splits=u"-", nosplit_re=r'\w_')

    def test_force_split_locations(self):

        self.assertListEqual(list(self.cc.force_split_locations("-hi")), [1])
        self.assertListEqual(list(self.cc.force_split_locations("a-hi")), [1,2])
        self.assertListEqual(list(self.cc.force_split_locations("a-")), [1])
        self.assertListEqual(list(self.cc.force_split_locations("a--")), [1,2])
        self.assertListEqual(list(self.cc.force_split_locations("a--h-")), [1,2,3,4])

        cc2 = BaseConstructionMethods(force_splits=u"-_")

        self.assertListEqual(list(cc2.force_split_locations("-_hi")), [1,2])
        self.assertListEqual(list(cc2.force_split_locations("a-_hi")), [1, 2, 3])
        self.assertListEqual(list(cc2.force_split_locations("a_hi")), [1, 2])

    def test_split_locations(self):
        c1 = u"hithere"
        c2 = u"hi_there"
        self.assertListEqual(list(self.cc.split_locations(c1)), list(range(1, 7)))

        self.assertListEqual(list(self.cc.split_locations(c1, start=0)), list(range(1, 7)))
        self.assertListEqual(list(self.cc.split_locations(c1, start=1)), list(range(2, 7)))
        self.assertListEqual(list(self.cc.split_locations(c1, stop=7)), list(range(1, 7)))
        self.assertListEqual(list(self.cc.split_locations(c1, stop=6)), list(range(1, 6)))

        self.assertListEqual(list(self.cc.split_locations(c2)), [1,3,4,5,6,7])
        self.assertListEqual(list(self.cc.split_locations(c2, start=1)), [3, 4, 5, 6, 7])
        self.assertListEqual(list(self.cc.split_locations(c2, start=1, stop=3)), [])


    def test_split(self):
        c1 = u"hithere"

        for false_split in [0,7]:
            self.assertRaises(AssertionError, self.cc.split, c1, false_split)

        self.assertEqual(tuple(self.cc.split(c1, 1)), (u"h", u"ithere"))

    def test_splitn(self):
        c1 = u"hithere"

        #false single splits
        for false_split in [0,7]:
            with self.assertRaises(AssertionError, msg="splitn with false_split {}".format(false_split)):
                list(self.cc.splitn(c1, false_split))

        #false iterable splits
        for false_split in [0,7]:
            with self.assertRaises(AssertionError, msg="splitn with false_split [{}]".format(false_split)):
                list(self.cc.splitn(c1, [false_split]))

        #empty split iterable
        self.assertEqual(tuple(self.cc.splitn(c1, [])), (c1,))

        #ok single split
        self.assertEqual(tuple(self.cc.splitn(c1, 1)), (u"h", u"ithere"))
        self.assertEqual(tuple(self.cc.splitn(c1, [2])), (u"hi", u"there"))

        #ok double split
        self.assertEqual(tuple(self.cc.splitn(c1, [2, 4])), (u"hi", u"th", u"ere"))

    def test_parts_to_splitloc(self):
        c1 = u"hithere"

        self.assertListEqual(list(self.cc.parts_to_splitlocs([])), [])
        self.assertListEqual(list(self.cc.parts_to_splitlocs([c1])), [])
        self.assertListEqual(list(self.cc.parts_to_splitlocs([c1, c1])), [7])
        self.assertListEqual(list(self.cc.parts_to_splitlocs([c1, c1, c1])), [7, 14])

    def test_slice(self):
        c1 = u"hithere"

        self.assertEqual(self.cc.slice(c1), c1)
        self.assertEqual(self.cc.slice(c1, start=0), c1)
        self.assertEqual(self.cc.slice(c1, stop=7), c1)

        self.assertEqual(self.cc.slice(c1, start=1), u"ithere")
        self.assertEqual(self.cc.slice(c1, stop=1), u"h")
        self.assertEqual(self.cc.slice(c1, start=2, stop=4), u"th")

    def test_from_string(self):
        self.assertEqual(self.cc.from_string(u"hithere"), u"hithere")

    def test_to_string(self):
        self.assertEqual(self.cc.to_string(u"hithere"), u"hithere")

    def test_corpus_key(self):
        self.assertEqual(self.cc.corpus_key(u"hithere"), u"hithere")

    def test_lex_key(self):
        self.assertEqual(self.cc.lex_key(u"hithere"), u"hithere")

    def test_atoms(self):
        self.assertEqual(self.cc.atoms(u"hithere"), u"hithere")