import json
import os
import pickle
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def fast_encode_model(model_name: str, chunk_texts: list, output_pkl: str, is_e5: bool = False, batch_size: int = 128, max_len: int = 384):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n--- Fast Encoding {len(chunk_texts)} chunks with {model_name} on {device} (FP16, batch_size={batch_size}, max_len={max_len}) ---", flush=True)

    if os.path.exists(output_pkl):
        print(f"File already exists: {output_pkl}", flush=True)
        return

    print("Loading HF Fast Tokenizer & Model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()

    if is_e5:
        prepared_texts = [f"passage: {t}" for t in chunk_texts]
    else:
        prepared_texts = chunk_texts

    n_samples = len(prepared_texts)
    embeddings_list = []

    t0 = time.time()
    with torch.inference_mode():
        for i in tqdm(range(0, n_samples, batch_size), desc=f"Encoding {model_name}"):
            batch_texts = prepared_texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=max_len, 
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            
            # Mean Pooling
            if hasattr(outputs, 'last_hidden_state'):
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = outputs[0][:, 0]

            # Normalize embeddings
            normalized = F.normalize(pooled, p=2, dim=1)
            embeddings_list.append(normalized.cpu().half().numpy())

    all_embeddings = np.vstack(embeddings_list)
    elapsed = time.time() - t0
    speed = n_samples / elapsed
    print(f"Finished {model_name} in {elapsed:.2f}s ({speed:.1f} chunks/sec). Output shape: {all_embeddings.shape}", flush=True)

    with open(output_pkl, 'wb') as f:
        pickle.dump(all_embeddings, f)
    print(f"Saved to {output_pkl}", flush=True)

    del model, tokenizer
    torch.cuda.empty_cache()

def main():
    base_dir = "d:/Study/DSC2026/LegalIR"
    cache_chunks_v2 = os.path.join(base_dir, "cache/article_chunks_v2.json")
    out_bgem3_emb = os.path.join(base_dir, "cache/bgem3_article_chunk_v2_embeddings.pkl")
    out_e5_emb = os.path.join(base_dir, "cache/e5_article_chunk_v2_embeddings.pkl")

    print(f"Loading Parent-Aware Article Chunks v2 from {cache_chunks_v2}...", flush=True)
    with open(cache_chunks_v2, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chunks = data['chunks']
    chunk_texts = [c['chunk_text'] for c in chunks]
    print(f"Loaded {len(chunk_texts)} chunks v2.", flush=True)

    # Fast encode BGE-M3
    fast_encode_model("BAAI/bge-m3", chunk_texts, out_bgem3_emb, is_e5=False, batch_size=128, max_len=384)

    # Fast encode E5-Large
    fast_encode_model("intfloat/multilingual-e5-large", chunk_texts, out_e5_emb, is_e5=True, batch_size=128, max_len=384)

if __name__ == "__main__":
    main()
