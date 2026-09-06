import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(r"D:\Study\DSC2026\LegalIR\src")))
from exp102_mil_nce_retrieval import (
    ResidualProjection,
    mil_document_score,
    compute_mil_nce_loss,
)


def test_residual_projection_identity_init():
    torch.manual_seed(42)
    model = ResidualProjection(dimension=1024, rank=32)
    model.eval()
    
    x = torch.randn(10, 1024)
    x = torch.nn.functional.normalize(x, p=2, dim=-1)
    
    with torch.no_grad():
        out = model(x)
        
    assert torch.allclose(out, x, atol=1e-6), "Residual projection must be exactly identity at initialization"
    print("test_residual_projection_identity_init: PASSED")


def test_mil_document_score_numerical_stability():
    torch.manual_seed(42)
    D = 1024
    N_chunks = 100
    
    query = torch.nn.functional.normalize(torch.randn(D), p=2, dim=-1)
    documents = torch.nn.functional.normalize(torch.randn(N_chunks, D), p=2, dim=-1)
    chunk_indices = torch.arange(N_chunks, dtype=torch.long)
    
    for tau in [1.0, 15.0, 50.0]:
        score = mil_document_score(query, chunk_indices, documents, tau_chunk=tau)
        assert not torch.isnan(score) and not torch.isinf(score), f"Score is NaN or Inf at tau={tau}"
        assert -1.2 <= score.item() <= 1.2, f"Score out of expected cosine range: {score.item()}"
    print("test_mil_document_score_numerical_stability: PASSED")


def test_multi_chunk_gradient_flow():
    """Verify that ALL chunks receive non-zero gradient during backward pass (no greedy hard-pruning)."""
    torch.manual_seed(42)
    D = 64
    N_chunks = 5
    
    query = torch.randn(D, requires_grad=True)
    query_norm = torch.nn.functional.normalize(query, p=2, dim=-1)
    
    docs_raw = torch.randn(N_chunks, D, requires_grad=True)
    docs_norm = torch.nn.functional.normalize(docs_raw, p=2, dim=-1)
    docs_norm.retain_grad()
    
    chunk_indices = torch.arange(N_chunks, dtype=torch.long)
    
    score = mil_document_score(query_norm, chunk_indices, docs_norm, tau_chunk=15.0)
    score.backward()
    
    doc_grads = docs_norm.grad
    assert doc_grads is not None, "Gradients must exist"
    grad_norms = torch.norm(doc_grads, dim=-1)
    for i, g_norm in enumerate(grad_norms):
        assert g_norm.item() > 1e-7, f"Chunk {i} received 0 gradient! Greedy trap detected."
    print("test_multi_chunk_gradient_flow: PASSED (All 5 chunks received active gradient!)")


if __name__ == "__main__":
    test_residual_projection_identity_init()
    test_mil_document_score_numerical_stability()
    test_multi_chunk_gradient_flow()
    print("\nALL UNIT TESTS FOR EXP-102 PASSED SUCCESSFULLY!")
