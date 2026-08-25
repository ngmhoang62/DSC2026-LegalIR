import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from exp025_both_model_miss_report import norm,tokens,ID
class Exp025Tests(unittest.TestCase):
 def test_folded_tokens_match_ascii_title(self): self.assertIn('thong',tokens('Thông tư'))
 def test_strict_identifier_ignores_dates(self): self.assertFalse(ID.search('ngày 01/7/2023'))
 def test_strict_identifier_accepts_citation(self): self.assertTrue(ID.search('Thông tư 12/2022/TT-BNV'))
if __name__=='__main__': unittest.main()
