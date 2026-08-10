import json
import glob
import os
import re
from tqdm import tqdm

def parse_document_into_articles(doc_json: dict) -> list:
    """
    Parses a single legal document JSON into Article-level chunks ('Điều X').
    Each chunk formatted as:
    {
        'chunk_id': 'doc_id_dieu_X',
        'doc_id': str(doc_id),
        'dieu_num': X,
        'title': doc_name,
        'chunk_text': f"{doc_name} - Điều {X}: {article_content}"
    }
    """
    doc_id = str(doc_json['id'])
    doc_name = doc_json.get('name', '').strip()
    passage = doc_json.get('passage', '').strip()

    if not passage:
        return [{
            'chunk_id': f"{doc_id}_full",
            'doc_id': doc_id,
            'dieu_num': 0,
            'title': doc_name,
            'chunk_text': doc_name
        }]

    # Regex to split passage by "Điều X"
    # Matches patterns like "Điều 1.", "Điều 2.", "Điều 12:"
    pattern = r'(\bĐiều\s+\d+[\.\:\s])'
    parts = re.split(pattern, passage)

    chunks = []

    # If no "Điều X" pattern found in passage, return full document as single chunk
    if len(parts) <= 1:
        chunks.append({
            'chunk_id': f"{doc_id}_full",
            'doc_id': doc_id,
            'dieu_num': 0,
            'title': doc_name,
            'chunk_text': f"{doc_name} {passage}".strip()
        })
        return chunks

    # First part before "Điều 1" (preamble / general info)
    preamble = parts[0].strip()
    if preamble:
        chunks.append({
            'chunk_id': f"{doc_id}_preamble",
            'doc_id': doc_id,
            'dieu_num': 0,
            'title': doc_name,
            'chunk_text': f"{doc_name} {preamble}".strip()
        })

    # Iterate over pairs of (header, content)
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        content = parts[i+1].strip() if i+1 < len(parts) else ""
        
        # Extract number X from "Điều X"
        dieu_match = re.search(r'\d+', header)
        dieu_num = int(dieu_match.group(0)) if dieu_match else i

        article_text = f"{header} {content}".strip()
        full_chunk_text = f"{doc_name} - {article_text}".strip() if doc_name else article_text

        chunks.append({
            'chunk_id': f"{doc_id}_dieu_{dieu_num}",
            'doc_id': doc_id,
            'dieu_num': dieu_num,
            'title': doc_name,
            'chunk_text': full_chunk_text
        })

    return chunks

def build_article_chunks_corpus(selected_contexts_dir: str, output_cache_path: str):
    """
    Processes all context JSON files and saves structured article chunks.
    """
    filepaths = glob.glob(os.path.join(selected_contexts_dir, "context_*.json"))
    all_chunks = []
    doc_to_chunk_ids = {}

    print(f"Parsing {len(filepaths)} legal documents into Article-level chunks...", flush=True)
    for fp in tqdm(filepaths):
        with open(fp, 'r', encoding='utf-8') as f:
            doc_json = json.load(f)
            chunks = parse_document_into_articles(doc_json)
            doc_id = str(doc_json['id'])
            
            all_chunks.extend(chunks)
            doc_to_chunk_ids[doc_id] = [c['chunk_id'] for c in chunks]

    result_payload = {
        "chunks": all_chunks,
        "doc_to_chunk_ids": doc_to_chunk_ids,
        "total_documents": len(filepaths),
        "total_chunks": len(all_chunks)
    }

    os.makedirs(os.path.dirname(output_cache_path), exist_ok=True)
    with open(output_cache_path, 'w', encoding='utf-8') as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved Article Chunking Corpus to {output_cache_path}")
    print(f"  Total Documents : {len(filepaths)}")
    print(f"  Total Chunks    : {len(all_chunks)}")
    print(f"  Mean Chunks/Doc : {len(all_chunks)/len(filepaths):.1f}")

if __name__ == "__main__":
    base_dir = "d:/Study/DSC2026/LegalIR"
    contexts_dir = os.path.join(base_dir, "public_test_dataset/selected-contexts")
    out_cache = os.path.join(base_dir, "cache/article_chunks.json")
    build_article_chunks_corpus(contexts_dir, out_cache)
