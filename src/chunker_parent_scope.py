import json
import glob
import os
import re
from tqdm import tqdm

def extract_chapter_and_scope(passage: str):
    """
    Extracts Scope (Điều 1) and Chapter markers from legal passage text.
    """
    # Extract Điều 1 scope if present
    scope_match = re.search(r'(Điều\s+1[\.\:\s][^\n\.]*(?:\n[^\n\.]+){0,3})', passage, re.IGNORECASE)
    scope_text = ""
    if scope_match:
        scope_text = scope_match.group(1).strip()
        # Cap scope length to 150 chars
        if len(scope_text) > 150:
            scope_text = scope_text[:150] + "..."

    return scope_text

def parse_document_into_articles_v2(doc_json: dict) -> list:
    """
    Parses a single legal document JSON into Parent-Aware Article-level chunks ('Điều X').
    Each chunk enriched with Chapter headers and Scope (Điều 1) context.
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
            'chapter': '',
            'scope': '',
            'chunk_text': doc_name if doc_name else doc_id
        }]

    scope_text = extract_chapter_and_scope(passage)

    # Regex to split passage by "Điều X"
    pattern = r'(\bĐiều\s+\d+[\.\:\s])'
    parts = re.split(pattern, passage)

    chunks = []

    # If no "Điều X" pattern found in passage, return full document as single chunk
    if len(parts) <= 1:
        full_text = f"{doc_name} {passage}".strip() if doc_name else passage
        chunks.append({
            'chunk_id': f"{doc_id}_full",
            'doc_id': doc_id,
            'dieu_num': 0,
            'title': doc_name,
            'chapter': '',
            'scope': scope_text,
            'chunk_text': full_text
        })
        return chunks

    # First part before "Điều 1" (preamble / general info)
    preamble = parts[0].strip()
    current_chapter = ""
    chapter_match = re.search(r'(Chương\s+[IVXLCDM\d]+[^\n\.\:]*)', preamble, re.IGNORECASE)
    if chapter_match:
        current_chapter = chapter_match.group(1).strip()

    if preamble:
        prefix_parts = []
        if doc_name:
            prefix_parts.append(doc_name)
        if current_chapter:
            prefix_parts.append(current_chapter)
        prefix = " - ".join(prefix_parts)
        chunk_str = f"{prefix}: {preamble}".strip() if prefix else preamble
        
        chunks.append({
            'chunk_id': f"{doc_id}_preamble",
            'doc_id': doc_id,
            'dieu_num': 0,
            'title': doc_name,
            'chapter': current_chapter,
            'scope': scope_text,
            'chunk_text': chunk_str
        })

    # Iterate over pairs of (header, content)
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        content = parts[i+1].strip() if i+1 < len(parts) else ""
        
        # Check if a new chapter is mentioned in content
        chapter_match = re.search(r'(Chương\s+[IVXLCDM\d]+[^\n\.\:]*)', content, re.IGNORECASE)
        if chapter_match:
            current_chapter = chapter_match.group(1).strip()

        # Extract number X from "Điều X"
        dieu_match = re.search(r'\d+', header)
        dieu_num = int(dieu_match.group(0)) if dieu_match else i

        article_text = f"{header} {content}".strip()
        
        # Construct Enriched Chunk Text
        prefix_components = []
        if doc_name:
            prefix_components.append(doc_name)
        if current_chapter:
            prefix_components.append(current_chapter)
        if scope_text and dieu_num != 1:
            prefix_components.append(f"Phạm vi: {scope_text}")

        if prefix_components:
            enriched_text = f"{' | '.join(prefix_components)} - {article_text}"
        else:
            enriched_text = article_text

        chunks.append({
            'chunk_id': f"{doc_id}_dieu_{dieu_num}",
            'doc_id': doc_id,
            'dieu_num': dieu_num,
            'title': doc_name,
            'chapter': current_chapter,
            'scope': scope_text,
            'chunk_text': enriched_text
        })

    return chunks

def build_article_chunks_v2_corpus(selected_contexts_dir: str, output_cache_path: str):
    """
    Processes all context JSON files and saves Parent-Aware article chunks (v2).
    """
    filepaths = glob.glob(os.path.join(selected_contexts_dir, "context_*.json"))
    all_chunks = []
    doc_to_chunk_ids = {}

    print(f"Parsing {len(filepaths)} legal documents into Parent-Aware Article chunks (v2)...", flush=True)
    for fp in tqdm(filepaths):
        with open(fp, 'r', encoding='utf-8') as f:
            doc_json = json.load(f)
            chunks = parse_document_into_articles_v2(doc_json)
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

    print(f"\nSaved Parent-Aware Article Chunks v2 Corpus to {output_cache_path}")
    print(f"  Total Documents : {len(filepaths)}")
    print(f"  Total Chunks    : {len(all_chunks)}")

if __name__ == "__main__":
    base_dir = "d:/Study/DSC2026/LegalIR"
    contexts_dir = os.path.join(base_dir, "public_test_dataset/selected-contexts")
    out_cache = os.path.join(base_dir, "cache/article_chunks_v2.json")
    build_article_chunks_v2_corpus(contexts_dir, out_cache)
