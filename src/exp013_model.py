"""Explicit model boundary for Jina ColBERT v2.

Remote model code is never executed unless the caller opts in.  This prevents a
normal preflight/audit from downloading or running arbitrary Hugging Face code.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def load_colbert_model(
    model_name: str,
    *,
    device: str = "cuda",
    allow_download: bool = False,
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer
    import torch

    if importlib.util.find_spec("einops") is None:
        raise RuntimeError(
            "Jina ColBERT v2 requires `einops`, which is now listed in requirements.txt. "
            "Install project dependencies with `python -m pip install -r requirements.txt`."
        )
    kwargs = {"local_files_only": not allow_download, "trust_remote_code": trust_remote_code}
    # Jina's 4.x remote config accesses deprecated Transformers 5 properties
    # internally.  Those notices cannot be solved by changing our arguments;
    # mute only that compatibility logger, not Python warnings globally.
    logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.integrations.tensor_parallel").setLevel(logging.ERROR)
    # The dynamically loaded Jina implementation logs one native-attention
    # fallback per transformer block on Windows.  It is expected and not an
    # actionable warning, so silence that module family only.
    logging.getLogger("transformers_modules.jinaai").setLevel(logging.ERROR)
    tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True, **kwargs)
    # Do not pass ``torch_dtype``/``dtype`` into from_pretrained here.  The
    # Jina remote configuration was authored for Transformers 4.x and treats
    # torch_dtype as an attribute name, whereas Transformers 5 passes a
    # torch.dtype instance.  Load from the checkpoint config, then cast the
    # instantiated model after configuration has completed.
    model = AutoModel.from_pretrained(model_name, **kwargs).to(device)
    if device.startswith("cuda"):
        model = model.half()
    model.eval()
    # The model card distributes the late-interaction projection separately as
    # ``linear.weight``. AutoModel correctly loads the XLM-R backbone but does
    # not know this ColBERT head, hence the prior "UNEXPECTED" load report.
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    snapshot = Path(snapshot_download(model_name, local_files_only=not allow_download))
    projection = load_file(str(snapshot / "model.safetensors"), device="cpu")["linear.weight"]
    model.colbert_projection = torch.nn.Linear(projection.shape[1], projection.shape[0], bias=False)
    model.colbert_projection.weight.data.copy_(projection)
    model.colbert_projection.to(device=device, dtype=torch.float16 if device.startswith("cuda") else torch.float32)
    model.colbert_projection.eval()
    return model, tokenizer


def colbert_encode(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    task: str,
    device: str,
    max_length: int = 384,
    dimensions: int = 64,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Normalize supported Jina output conventions into per-text token arrays."""
    import torch

    marker = "[QueryMarker] " if task == "retrieval.query" else "[DocumentMarker] "
    marked_texts = [marker + str(text) for text in texts]
    encoded = tokenizer(marked_texts, padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt", return_attention_mask=True)
    token_ids = encoded["input_ids"].cpu().numpy()
    masks = encoded["attention_mask"].cpu().numpy()
    with torch.inference_mode():
        output = model(**{name: value.to(device) for name, value in encoded.items()}, return_dict=True)
        hidden = getattr(output, "last_hidden_state", output[0])
        projected = model.colbert_projection(hidden)[..., :dimensions]
        projected = torch.nn.functional.normalize(projected.float(), p=2, dim=-1).cpu().numpy()
    if projected.ndim != 3 or projected.shape[0] != len(texts):
        raise RuntimeError(
            "Jina model did not return [batch, tokens, dim] token embeddings. "
            "The XLM-R backbone could not be projected through the ColBERT head."
        )
    if projected.shape[-1] != dimensions:
        raise RuntimeError(
            f"Late-interaction dimension mismatch: expected {dimensions}, got {values.shape[-1]}. "
            "The configured model does not support the selected Matryoshka dimension."
        )
    vectors: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    active: list[np.ndarray] = []
    for row in range(len(texts)):
        # Remove XLM-R special tokens; preserve the task marker, whose learned
        # vector is needed for asymmetric query/passage encoding.
        valid = np.flatnonzero(masks[row]).astype(np.int64)
        specials = set(getattr(tokenizer, "all_special_ids", ()))
        valid = np.asarray([index for index in valid if int(token_ids[row, index]) not in specials], dtype=np.int64)
        vectors.append(np.asarray(projected[row, valid], dtype=np.float32))
        ids.append(np.asarray(token_ids[row, valid], dtype=np.int64))
        active.append(np.ones(len(valid), dtype=np.int8))
    return vectors, ids, active
