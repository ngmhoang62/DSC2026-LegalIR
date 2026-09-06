from __future__ import annotations

import pytest
import torch

from exp_final.learning import query_encoder_contract


def test_e5_contract_is_unchanged():
    value = query_encoder_contract("e5")
    assert value["repo"] == "mainguyen9/vietlegal-e5"
    assert value["prefix"] == "query: "
    assert value["pooling"] == "mean"
    assert value["target_modules"] == ["query", "value"]
    assert value["dtype"] == torch.float32


def test_lal_contract_preserves_native_encoder_semantics():
    value = query_encoder_contract("lal")
    assert value["repo"] == "darklethelong/vnlegal-lal"
    assert value["prefix"].endswith("\nQuery: ")
    assert value["max_length"] == 2048
    assert value["pooling"] == "last_non_padding"
    assert value["target_modules"] == ["q_proj", "v_proj"]
    assert value["dtype"] == torch.float16


def test_unknown_query_source_rejected():
    with pytest.raises(ValueError):
        query_encoder_contract("unknown")
