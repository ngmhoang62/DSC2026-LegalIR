import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC=Path(__file__).resolve().parents[1]/"src"; sys.path.insert(0,str(SRC))
from exp027_lambdamart_shortlist import K_GRID, retained_answers

class Exp027Tests(unittest.TestCase):
    def test_grid_reaches_100(self):
        self.assertEqual(K_GRID, (16,24,32,50,64,80,100))

    def test_retained_answers_match_impact(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); train=root/'train.json'; exclusions=root/'exclusions.json'; impact=root/'impact.jsonl'
            train.write_text(json.dumps({'q':{'answer':['a','b']}}),encoding='utf8'); exclusions.write_text(json.dumps([{'doc_id':'a'}]),encoding='utf8'); impact.write_text(json.dumps({'intentionally_excluded_gold_ids':['a']})+'\n',encoding='utf8')
            answers,stats=retained_answers(train,exclusions,impact)
            self.assertEqual(answers['q'],{'b'}); self.assertEqual(stats['removed_gold_occurrences'],1)

if __name__=='__main__': unittest.main()
