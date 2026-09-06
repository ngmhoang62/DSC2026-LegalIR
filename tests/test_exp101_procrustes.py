import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(r"D:\Study\DSC2026\LegalIR\src")))
from exp101_procrustes_alignment import (
    l2_normalize,
    solve_orthogonal_procrustes,
    solve_ridge_alignment,
)


def test_l2_normalize():
    vecs = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    normed = l2_normalize(vecs)
    assert np.allclose(np.linalg.norm(normed, axis=1), [1.0, 1.0], atol=1e-6)
    print("test_l2_normalize: PASSED")


def test_orthogonal_procrustes_properties():
    np.random.seed(42)
    N, D = 100, 16
    X = np.random.randn(N, D).astype(np.float32)
    R, _ = np.linalg.qr(np.random.randn(D, D))
    Y = X @ R

    W = solve_orthogonal_procrustes(X, Y)

    # 1. W must be orthogonal: W^T @ W = I
    I = np.eye(D, dtype=np.float32)
    assert np.allclose(W.T @ W, I, atol=1e-5), "W must be orthogonal"
    assert np.allclose(W @ W.T, I, atol=1e-5), "W must be orthogonal"

    # 2. Distance preservation (Norm preservation)
    x_test = np.random.randn(10, D).astype(np.float32)
    x_test_norm = np.linalg.norm(x_test, axis=1)
    rot_norm = np.linalg.norm(x_test @ W, axis=1)
    assert np.allclose(x_test_norm, rot_norm, atol=1e-5), "Rotation must preserve vector norms"

    # 3. Recovery of the synthetic rotation
    assert np.allclose(X @ W, Y, atol=1e-4), "Procrustes should accurately align rotated subspace"
    print("test_orthogonal_procrustes_properties: PASSED")


def test_ridge_alignment():
    np.random.seed(42)
    N, D = 100, 16
    X = np.random.randn(N, D).astype(np.float32)
    True_W = np.random.randn(D, D).astype(np.float32)
    Y = X @ True_W + 0.01 * np.random.randn(N, D).astype(np.float32)

    W_ridge = solve_ridge_alignment(X, Y, lambda_reg=1e-3)
    assert W_ridge.shape == (D, D)
    reconstruction = l2_normalize(X) @ W_ridge
    # Check that solved ridge is close to True_W / scale
    err = np.mean((l2_normalize(X) @ W_ridge - l2_normalize(Y)) ** 2)
    assert err < 0.2
    print("test_ridge_alignment: PASSED")


if __name__ == "__main__":
    test_l2_normalize()
    test_orthogonal_procrustes_properties()
    test_ridge_alignment()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
