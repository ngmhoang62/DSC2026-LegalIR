import json
import glob
import os
import re
import numpy as np
from tqdm import tqdm

def analyze_dataset():
    base_dir = "d:/Study/DSC2026/LegalIR"
    train_file = os.path.join(base_dir, "public_test_dataset/train.json")
    public_file = os.path.join(base_dir, "public_test_dataset/public-official.json")
    selected_contexts_dir = os.path.join(base_dir, "public_test_dataset/selected-contexts")
    out_eda_file = os.path.join(base_dir, "results/eda_summary.txt")

    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(public_file, 'r', encoding='utf-8') as f:
        public_data = json.load(f)

    lines = []
    lines.append("==========================================================")
    lines.append("1. QUERY ANALYSIS (Train & Public)")
    lines.append("==========================================================")

    train_queries = [v['question'] for v in train_data.values()]
    public_queries = [v['question'] for v in public_data.values()]

    def query_stats(queries, name):
        word_lens = [len(q.split()) for q in queries]
        char_lens = [len(q) for q in queries]
        lines.append(f"\n--- {name} ({len(queries)} queries) ---")
        lines.append(f"Word count  : Min={np.min(word_lens)}, Max={np.max(word_lens)}, Mean={np.mean(word_lens):.1f}, Median={np.median(word_lens):.1f}, P95={np.percentile(word_lens, 95):.1f}")
        lines.append(f"Char count  : Min={np.min(char_lens)}, Max={np.max(char_lens)}, Mean={np.mean(char_lens):.1f}, Median={np.median(char_lens):.1f}")

        # Intent / Keyword patterns
        has_dieu = sum(1 for q in queries if re.search(r'\bđiều\b', q, re.I))
        has_khoan = sum(1 for q in queries if re.search(r'\bkhoản\b', q, re.I))
        has_nghi_dinh = sum(1 for q in queries if re.search(r'nghị định|thông tư|quyết định|luật|nghị quyết|bộ luật', q, re.I))
        has_number_code = sum(1 for q in queries if re.search(r'\d+/\d+', q))
        has_year = sum(1 for q in queries if re.search(r'\b(201\d|202\d)\b', q))

        lines.append(f"  - Mentions 'Điều'          : {has_dieu} ({has_dieu/len(queries)*100:.1f}%)")
        lines.append(f"  - Mentions 'Khoản'         : {has_khoan} ({has_khoan/len(queries)*100:.1f}%)")
        lines.append(f"  - Mentions Law/Decree Type : {has_nghi_dinh} ({has_nghi_dinh/len(queries)*100:.1f}%)")
        lines.append(f"  - Mentions Legal Code Num  : {has_number_code} ({has_number_code/len(queries)*100:.1f}%)")
        lines.append(f"  - Mentions Year (201x-202x): {has_year} ({has_year/len(queries)*100:.1f}%)")

    query_stats(train_queries, "Train Queries")
    query_stats(public_queries, "Public Test Queries")

    lines.append("\n==========================================================")
    lines.append("2. CORPUS / LEGAL DOCUMENT ANALYSIS (8,532 Documents)")
    lines.append("==========================================================")

    filepaths = glob.glob(os.path.join(selected_contexts_dir, "context_*.json"))
    
    doc_word_lens = []
    doc_char_lens = []
    
    has_dieu_count = 0
    has_khoan_count = 0
    has_chuong_count = 0
    has_subclauses = 0  # a), b), c)
    has_title_name = 0

    dieu_counts_per_doc = []

    for fp in tqdm(filepaths, desc="Analyzing documents"):
        with open(fp, 'r', encoding='utf-8') as f:
            doc = json.load(f)
            name = doc.get('name', '')
            passage = doc.get('passage', '')
            full_text = f"{name} {passage}".strip()

            words = full_text.split()
            doc_word_lens.append(len(words))
            doc_char_lens.append(len(full_text))

            if name:
                has_title_name += 1

            # Search structure
            dieu_matches = re.findall(r'\bĐiều\s+\d+', passage)
            if dieu_matches:
                has_dieu_count += 1
                dieu_counts_per_doc.append(len(dieu_matches))
            else:
                dieu_counts_per_doc.append(0)

            if re.search(r'\bkhoản\s+\d+\b', passage, re.I):
                has_khoan_count += 1
            if re.search(r'\bchương\s+[I|V|X|L|C|D|M]+\b', passage, re.I):
                has_chuong_count += 1
            if re.search(r'\b[a-đ]\)\s', passage):
                has_subclauses += 1

    total_docs = len(filepaths)
    lines.append(f"\nDocument Count : {total_docs}")
    lines.append(f"Word Length    : Min={np.min(doc_word_lens)}, Max={np.max(doc_word_lens)}, Mean={np.mean(doc_word_lens):.1f}, Median={np.median(doc_word_lens):.1f}")
    lines.append(f"Char Length    : Min={np.min(doc_char_lens)}, Max={np.max(doc_char_lens)}, Mean={np.mean(doc_char_lens):.1f}, Median={np.median(doc_char_lens):.1f}")
    lines.append(f"  - Words <= 512 words : {sum(1 for w in doc_word_lens if w <= 512)} ({sum(1 for w in doc_word_lens if w <= 512)/total_docs*100:.1f}%)")
    lines.append(f"  - Words > 512 words  : {sum(1 for w in doc_word_lens if w > 512)} ({sum(1 for w in doc_word_lens if w > 512)/total_docs*100:.1f}%)")
    lines.append(f"  - Words > 1024 words : {sum(1 for w in doc_word_lens if w > 1024)} ({sum(1 for w in doc_word_lens if w > 1024)/total_docs*100:.1f}%)")
    lines.append(f"  - Words > 2048 words : {sum(1 for w in doc_word_lens if w > 2048)} ({sum(1 for w in doc_word_lens if w > 2048)/total_docs*100:.1f}%)")

    lines.append(f"\nDocument Structural Traits:")
    lines.append(f"  - Docs with 'name' (Title)        : {has_title_name} ({has_title_name/total_docs*100:.1f}%)")
    lines.append(f"  - Docs containing 'Điều X'        : {has_dieu_count} ({has_dieu_count/total_docs*100:.1f}%)")
    lines.append(f"  - Docs containing 'Khoản Y'       : {has_khoan_count} ({has_khoan_count/total_docs*100:.1f}%)")
    lines.append(f"  - Docs containing 'Chương Z'      : {has_chuong_count} ({has_chuong_count/total_docs*100:.1f}%)")
    lines.append(f"  - Docs containing sub-items a),b): {has_subclauses} ({has_subclauses/total_docs*100:.1f}%)")

    lines.append(f"\nArticles ('Điều') per Document distribution:")
    dieu_non_zero = [c for c in dieu_counts_per_doc if c > 0]
    lines.append(f"  - Mean 'Điều' per doc (where present): {np.mean(dieu_non_zero):.1f}" if dieu_non_zero else "0")
    lines.append(f"  - Max 'Điều' in single doc           : {np.max(dieu_counts_per_doc)}")

    os.makedirs(os.path.dirname(out_eda_file), exist_ok=True)
    with open(out_eda_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Saved EDA report to {out_eda_file}")

if __name__ == "__main__":
    analyze_dataset()
