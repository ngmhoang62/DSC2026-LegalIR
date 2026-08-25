import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC=Path(__file__).resolve().parents[1]/'src'; sys.path.insert(0,str(SRC))
from exp026_lambdamart_capsules import _columns, _orders


class Exp026Tests(unittest.TestCase):
    def test_feature_blocks_are_additive(self):
        self.assertEqual(_columns('retrieval'), ('candidate_rank','e5_rank','e5_score','e5_recip','bm25_rank','bm25_recip'))
        self.assertGreater(len(_columns('all')), len(_columns('retrieval+provenance')))

    def test_orders_are_deterministic_and_keep_pool(self):
        row={'candidates':[{'doc_id':'b','rank':1,'sources':{'e5':{'rank':2,'aggregate_score':.2}}},{'doc_id':'a','rank':2,'sources':{'bm25':{'rank':1,'passage_ranks':[1]}}},{'doc_id':'c','rank':3,'sources':{'e5':{'rank':1,'aggregate_score':.3},'bm25':{'rank':2,'passage_ranks':[3]}}}]}
        orders=_orders(row)
        self.assertEqual(orders['union'],['b','a','c'])
        self.assertEqual(orders['e5_rank'],['c','b','a'])
        self.assertEqual(orders['bm25_rank'],['a','c','b'])
        self.assertEqual(set(orders['e5_rank']),set(orders['union']))

if __name__=='__main__': unittest.main()
