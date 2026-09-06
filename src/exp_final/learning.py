from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import ROOT, CACHE, RESULTS, Progress, digest, read, seed_all, sha, write


def multi_loss(positives, negatives, temperature=.05):
    if not len(positives) or not len(negatives):
        raise ValueError("Both positive and negative scores required")
    p, n = positives / temperature, negatives / temperature
    return (torch.logaddexp(p, torch.logsumexp(n, dim=0)) - p).mean()


def case_multi_loss(anchor, positives, negatives, temperature=.05):
    """Multi-positive query-to-query contrastive loss.

    ``anchor`` is one normalized query vector. Positive support queries share
    at least one supplied parent label with the anchor; negatives share none.
    Keeping the denominator free of other positives mirrors ``multi_loss`` and
    prevents two queries for the same legal parent from competing.
    """
    if anchor.ndim != 1 or positives.ndim != 2 or negatives.ndim != 2:
        raise ValueError("case_multi_loss expects [D], [P,D], [N,D]")
    if not len(positives) or not len(negatives):
        raise ValueError("Both positive and negative case vectors required")
    return multi_loss(positives @ anchor, negatives @ anchor, temperature)


def boundary_loss(p, n, pranks, nranks, temperature=.05):
    mask = (pranks[:, None] <= 5) != (nranks[None, :] <= 5)
    terms = F.softplus((n[None, :] - p[:, None]) / temperature) * mask
    return (terms.sum(1) / mask.sum(1).clamp_min(1)).mean()


def ce_loss(logits, positive_count):
    p, n = logits[:positive_count], logits[positive_count:]
    balanced = .5 * (F.softplus(-p).mean() + F.softplus(n).mean())
    return multi_loss(p, n, 1.) + .1 * balanced


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def replay_backward(forward, batches, positive_count, scale=1.):
    """Exact first-order scalar-logit replay, including stochastic dropout."""
    states, logits = [], []
    with torch.no_grad():
        for batch in batches:
            states.append(rng_state())
            logits.append(forward(batch).reshape(-1))
    terminal_rng = rng_state()
    leaf = torch.cat(logits).detach().requires_grad_(True)
    loss = ce_loss(leaf, positive_count) * scale
    derivative, = torch.autograd.grad(loss, leaf)
    offset = 0
    for batch, state in zip(batches, states):
        set_rng(state)
        value = forward(batch).reshape(-1)
        value.backward(derivative[offset:offset + len(value)])
        offset += len(value)
    set_rng(terminal_rng)
    return float(loss.detach())


def select_negatives(current, gold, sources, universe, qid, epoch, count=64):
    rng = random.Random(int(digest([qid, epoch, 112])[:16], 16))
    blocked, selected = set(gold), []
    def add(values, n):
        found = 0
        for d in values:
            if d not in blocked:
                selected.append(d)
                blocked.add(d)
                found += 1
                if found >= n:
                    break
    if count == 6:
        add(current, 2)
        middle = list(current[8:32]); rng.shuffle(middle); add(middle, 2)
        add(sources.get("bm25", []) + sources.get("lal", []), 1)
        deep = list(current[32:]) or list(universe); rng.shuffle(deep); add(deep, 1)
    else:
        add(current, 16)
        middle = list(current[16:100]); rng.shuffle(middle); add(middle, 16)
        disagreement = []
        base = set(current[:16])
        for items in zip(sources.get("bm25", []), sources.get("lal", [])):
            disagreement.extend(d for d in items if d not in base)
        rng.shuffle(disagreement); add(disagreement, 16)
        add(sources.get("jina", []) or sources.get("e5", []), 8)
        randoms = list(universe); rng.shuffle(randoms); add(randoms, 8)
    if len(selected) < count:
        add(current, count-len(selected))
    if len(selected) < count:
        rest = list(universe); rng.shuffle(rest); add(rest, count-len(selected))
    if len(selected) != count:
        raise ValueError("Not enough unique negatives")
    return selected


class ParentBank:
    def __init__(self, vectors, parent_indices, *, device="cuda"):
        self.vectors = torch.empty(tuple(vectors.shape), dtype=torch.float32, device=device)
        # Copy/normalize bounded blocks: no read-only mmap alias or second full bank.
        for start in range(0, len(self.vectors), 8192):
            block = self.vectors[start:start+8192]
            if torch.is_tensor(vectors):
                block.copy_(vectors[start:start+8192].detach())
            else:
                block.copy_(torch.from_numpy(np.array(vectors[start:start+8192], dtype=np.float32, copy=True)))
            block.copy_(F.normalize(block, dim=1))
        self.parent = torch.as_tensor(parent_indices, dtype=torch.long, device=device)
        self.count = int(self.parent.max()) + 1
        self.counts = torch.bincount(self.parent, minlength=self.count)
        self.chunk_ids = torch.arange(len(self.parent), device=device)

    @torch.no_grad()
    def mine(self, query):
        scores = query.detach().float() @ self.vectors.T
        if scores.ndim == 1:
            scores = scores[None]
        b = len(scores)
        ids = self.parent.expand(b, -1)
        first = torch.full((b, self.count), -torch.inf, device=scores.device)
        first.scatter_reduce_(1, ids, scores, reduce="amax", include_self=True)
        sentinel = len(self.parent)
        arg1 = torch.full((b, self.count), sentinel, device=scores.device, dtype=torch.long)
        eligible = torch.where(scores == first.gather(1, ids), self.chunk_ids, sentinel)
        arg1.scatter_reduce_(1, ids, eligible, reduce="amin", include_self=True)
        rest = scores.masked_fill(self.chunk_ids[None] == arg1.gather(1, ids), -torch.inf)
        second = torch.full_like(first, -torch.inf)
        second.scatter_reduce_(1, ids, rest, reduce="amax", include_self=True)
        arg2 = torch.full_like(arg1, sentinel)
        eligible2 = torch.where(rest == second.gather(1, ids), self.chunk_ids, sentinel)
        arg2.scatter_reduce_(1, ids, eligible2, reduce="amin", include_self=True)
        singleton = self.counts == 1
        arg2[:, singleton] = arg1[:, singleton]
        result = torch.where(singleton, first, .5 * (first + second))
        return result, torch.stack((arg1, arg2), dim=-1)

    def rescore(self, query, selected_indices):
        return (self.vectors[selected_indices] * query.float()).sum(-1).mean(-1)


def local_snapshot(repo):
    roots = [Path(os.environ.get("HF_HOME", "C:/Users/nguye/.cache/huggingface")) / "hub",
             ROOT.parent / "models", ROOT / "models"]
    for root in roots:
        folder = root / ("models--" + repo.replace("/", "--"))
        ref = folder / "refs/main"
        candidates = ([folder / "snapshots" / ref.read_text().strip()] if ref.exists() else [])
        candidates += sorted((folder / "snapshots").glob("*"))
        for p in candidates:
            if (p / "config.json").exists() and (p / "tokenizer_config.json").exists():
                return p
    raise FileNotFoundError(f"Pinned local model unavailable: {repo}")


QUERY_ENCODER_CONTRACTS = {
    "e5": {
        "repo": "mainguyen9/vietlegal-e5",
        "prefix": "query: ",
        "max_length": 512,
        "pooling": "mean",
        "target_modules": ["query", "value"],
        "dtype": torch.float32,
    },
    "lal": {
        "repo": "darklethelong/vnlegal-lal",
        "prefix": "Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: ",
        "max_length": 2048,
        "pooling": "last_non_padding",
        "target_modules": ["q_proj", "v_proj"],
        "dtype": torch.float16,
    },
}


def query_encoder_contract(source="e5"):
    if source not in QUERY_ENCODER_CONTRACTS:
        raise ValueError(f"Unsupported query encoder source: {source}")
    return dict(QUERY_ENCODER_CONTRACTS[source])


def serializable_query_encoder_contract(source="e5"):
    value = query_encoder_contract(source)
    value["dtype"] = str(value["dtype"])
    return value


class QueryEncoder(nn.Module):
    def __init__(self, device="cuda", dtype=None, checkpoint=None, source="e5"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        self.source = source
        self.contract = query_encoder_contract(source)
        dtype = dtype or self.contract["dtype"]
        self.snapshot = local_snapshot(self.contract["repo"])
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot), local_files_only=True)
        base = AutoModel.from_pretrained(str(self.snapshot), local_files_only=True, dtype=dtype)
        self.model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                                     target_modules=self.contract["target_modules"], bias="none"))
        # Keep the larger Qwen/LAL frozen tower in FP16 while retaining FP32
        # adapter weights and optimizer state. PEFT casts activations at the
        # LoRA boundary and returns the update in the base layer's dtype.
        if source == "lal":
            for parameter in self.model.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        self.to(device)
        if checkpoint:
            self.load_adapter(checkpoint)

    def forward(self, texts):
        batch = self.tokenizer([self.contract["prefix"] + t for t in texts], padding=True, truncation=True,
                               max_length=self.contract["max_length"], return_tensors="pt").to(next(self.parameters()).device)
        hidden = self.model(**batch).last_hidden_state.float()
        if self.contract["pooling"] == "mean":
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        else:
            mask = batch["attention_mask"].bool()
            if (~mask.any(dim=1)).any():
                raise ValueError("Cannot last-token pool an all-padding query")
            positions = mask.shape[1] - 1 - torch.flip(mask, dims=[1]).long().argmax(dim=1)
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        return F.normalize(pooled, dim=-1)

    def adapter_state(self):
        return {k: v.detach().cpu() for k, v in self.named_parameters() if v.requires_grad}

    def load_adapter(self, path):
        value = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(value["adapter"], strict=False)


def checkpoint(path, model, optimizer, scheduler, **extra):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(adapter=model.adapter_state(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(), **extra), tmp)
    os.replace(tmp, path)
    write(path.with_suffix('.sha.json'), dict(sha256=sha(path)))


def train_query(data, store, qids, directory, epochs=2, nominal_epochs=2, microbatch=1,
                max_updates=None, positive_policy="all", learning_rate=5e-5, source="e5"):
    from transformers import get_cosine_schedule_with_warmup
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    qids = [q for q in qids if data.gold[q]]
    if positive_policy == "single_only":
        qids = [q for q in qids if len(data.gold[q]) == 1]
    elif positive_policy not in ("all", "content_primary"):
        raise ValueError(f"Unknown positive policy: {positive_policy}")
    contract = dict(qids=qids, epochs=epochs, nominal_epochs=nominal_epochs,
                    microbatch=microbatch, data=data.fingerprint,
                    positive_policy=positive_policy, learning_rate=learning_rate, source=source,
                    encoder_contract=serializable_query_encoder_contract(source))
    if not max_updates:
        contract['code_sha256'] = sha(Path(__file__))
    contract_hash = digest(contract)
    benchmark_path = directory / "BENCHMARK.json"
    if max_updates and benchmark_path.exists():
        value = read(benchmark_path)
        if value["contract"] != contract_hash or value["updates"] != max_updates:
            raise ValueError("Benchmark contract changed")
        return value
    done = directory / "_SUCCESS.json"
    if done.exists():
        if read(done)["contract"] != contract_hash:
            raise ValueError("Query training resume contract changed")
        if any(sha(directory/name) != expected for name, expected in read(done)["files"].items()):
            raise ValueError("Query completed checkpoint hash mismatch")
        return read(done)
    seed_all()
    model = QueryEncoder(source=source)
    bank = data.bank(source, device="cuda")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=.01)
    steps = math.ceil(len(qids) / 16)
    scheduler = get_cosine_schedule_with_warmup(opt, max(1, int(.1 * steps * nominal_epochs)), steps * nominal_epochs)
    state_path = directory / "resume.pt"
    start_epoch, start_pos, updates = 0, 0, 0
    if state_path.exists():
        receipt = state_path.with_suffix('.sha.json')
        if not receipt.exists() or read(receipt)['sha256'] != sha(state_path):
            raise ValueError("Query resume checkpoint lacks verified hash")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["contract"] != contract_hash:
            raise ValueError("Query checkpoint contract changed")
        model.load_state_dict(state["adapter"], strict=False)
        opt.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        set_rng(state["rng"])
        start_epoch, start_pos, updates = state["epoch"], state["position"], state["updates"]
    progress = Progress("query_" + directory.name, epochs * len(qids))
    import time
    begin = time.monotonic()
    try:
        for epoch in range(start_epoch, epochs):
            order = list(qids); random.Random(112 + epoch).shuffle(order)
            model.train()
            for pos in range(start_pos if epoch == start_epoch else 0, len(order), 16):
                group = order[pos:pos+16]; opt.zero_grad(set_to_none=True)
                loss_total = 0.
                for start in range(0, len(group), microbatch):
                    ids = group[start:start+microbatch]
                    qvec = model([data.questions[q] for q in ids])
                    scores, indices = bank.mine(qvec)
                    ordering = torch.argsort(scores, dim=1, descending=True, stable=True)
                    ranks = torch.argsort(ordering, dim=1, stable=True) + 1
                    local_loss = []
                    for b, q in enumerate(ids):
                        current = [data.doc_ids[i] for i in ordering[b].cpu().tolist()]
                        src = store.rankings(q)
                        negative = select_negatives(current, data.gold[q], src, data.doc_ids, q, epoch)
                        positives = sorted(data.gold[q])
                        if positive_policy == "content_primary" and len(positives) > 1:
                            candidate_rows = torch.tensor([data.doc_row[d] for d in positives], device="cuda")
                            # Selection is intentionally discrete. Gradient still
                            # flows through the selected parent score below.
                            selected = int(torch.argmax(scores[b, candidate_rows]).item())
                            positives = [positives[selected]]
                        pi = torch.tensor([data.doc_row[d] for d in positives], device="cuda")
                        ni = torch.tensor([data.doc_row[d] for d in negative], device="cuda")
                        p = bank.rescore(qvec[b], indices[b, pi]); n = bank.rescore(qvec[b], indices[b, ni])
                        loss = multi_loss(p, n)
                        if epoch > 0:
                            loss = loss + .25 * boundary_loss(p, n, ranks[b, pi], ranks[b, ni])
                        frozen = torch.as_tensor(data.query_vector(q, source), device="cuda")
                        loss = loss + .05 * (1 - F.cosine_similarity(qvec[b:b+1], frozen[None]).mean())
                        local_loss.append(loss)
                    batch_loss = torch.stack(local_loss).sum() / len(group)
                    if not torch.isfinite(batch_loss):
                        raise FloatingPointError("Non-finite query loss")
                    batch_loss.backward(); loss_total += float(batch_loss.detach())
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                if float(gradient_norm) == 0.:
                    raise ValueError("Query backbone received no nonzero gradients")
                opt.step(); scheduler.step(); updates += 1
                if updates % 16 == 0 or pos + len(group) == len(order):
                    checkpoint(state_path, model, opt, scheduler, contract=contract_hash, epoch=epoch, position=pos+len(group), updates=updates)
                progress.update(epoch * len(qids) + pos + len(group), loss=loss_total, gradient_norm=float(gradient_norm))
                if max_updates and updates >= max_updates:
                    checkpoint(state_path, model, opt, scheduler, contract=contract_hash, epoch=epoch, position=pos+len(group), updates=updates)
                    value = {"contract": contract_hash, "benchmark_only": True, "updates": updates, "seconds": time.monotonic()-begin,
                             "seconds_per_update": (time.monotonic()-begin)/updates, "peak_vram": torch.cuda.max_memory_reserved()}
                    write(benchmark_path, value)
                    return value
            checkpoint(directory / f"epoch-{epoch+1}.pt", model, opt, scheduler, contract=contract_hash, epoch=epoch+1, position=0, updates=updates)
        result = {"contract": contract_hash, "training_qids": qids, "epochs": epochs, "updates": updates,
                  "seconds": time.monotonic()-begin, "files": {p.name: sha(p) for p in directory.glob("epoch-*.pt")}}
        write(done, result); return result
    finally:
        del model, bank, opt, scheduler
        import gc
        gc.collect(); torch.cuda.empty_cache()


class CaseSupportSampler:
    """Fold-scoped semantic case sampler built only from training labels.

    Frozen LAL similarity chooses *which* labeled peer/confuser to expose. The
    trainable encoder supplies every score used by the case objective. This is
    deliberately a training sampler, never an inference-time label feature.
    """
    def __init__(self, data, qids, *, hard_pool=64):
        from collections import defaultdict
        self.qids = [str(q) for q in qids if data.gold.get(str(q))]
        self.row = {q: i for i, q in enumerate(self.qids)}
        self.labels = {q: set(data.gold[q]) for q in self.qids}
        vectors = np.stack([data.query_vector(q, "lal") for q in self.qids]).astype(np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        similarities = np.asarray(vectors @ vectors.T, dtype=np.float32)
        by_doc = defaultdict(list)
        for q in self.qids:
            for doc in self.labels[q]:
                by_doc[doc].append(self.row[q])
        self.positives, self.negatives = {}, {}
        all_rows = np.arange(len(self.qids))
        for i, q in enumerate(self.qids):
            positive_rows = set()
            for doc in self.labels[q]:
                positive_rows.update(by_doc[doc])
            positive_rows.discard(i)
            self.positives[q] = sorted(
                (self.qids[j] for j in positive_rows),
                key=lambda candidate: (-float(similarities[i, self.row[candidate]]), candidate),
            )
            blocked = np.zeros(len(self.qids), dtype=bool)
            blocked[i] = True
            for doc in self.labels[q]:
                blocked[np.asarray(by_doc[doc], dtype=np.int64)] = True
            eligible = all_rows[~blocked]
            take = min(int(hard_pool), len(eligible))
            if not take:
                raise ValueError(f"No disjoint-label case negatives for {q}")
            local = similarities[i, eligible]
            chosen = eligible[np.argpartition(-local, take - 1)[:take]]
            self.negatives[q] = sorted(
                (self.qids[j] for j in chosen),
                key=lambda candidate: (-float(similarities[i, self.row[candidate]]), candidate),
            )
        self.eligible = [q for q in self.qids if self.positives[q]]
        # Release the O(N^2) construction matrix before GPU training starts.
        del similarities, vectors

    def sample(self, qid, epoch, *, positive_count=1, negative_count=4):
        qid = str(qid)
        positives = self.positives.get(qid, [])
        if not positives:
            return [], []
        seed = int(digest([qid, epoch, 112, "case-support-v1"])[:16], 16)
        pstart = seed % min(4, len(positives))
        selected_p = [positives[(pstart + i) % min(4, len(positives))]
                      for i in range(min(positive_count, len(positives)))]
        negatives = self.negatives[qid]
        nwindow = min(32, len(negatives))
        nstart = (seed // 17 + epoch * negative_count) % nwindow
        selected_n = [negatives[(nstart + i) % nwindow] for i in range(negative_count)]
        if set(selected_p) & set(selected_n):
            raise AssertionError("Case positive leaked into negatives")
        if any(self.labels[qid] & self.labels[q] for q in selected_n):
            raise AssertionError("Shared-label query selected as case negative")
        return selected_p, selected_n


def train_query_with_case(data, store, qids, directory, *, epochs=2, nominal_epochs=2,
                          max_updates=None, learning_rate=5e-5, case_weight=.2,
                          case_positive_count=1, case_negative_count=4):
    """Adapt the LAL query tower jointly for parent retrieval and case memory."""
    from transformers import get_cosine_schedule_with_warmup
    import time
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    qids = [str(q) for q in qids if data.gold.get(str(q))]
    sampler = CaseSupportSampler(data, qids)
    contract = {
        "qids": qids, "epochs": epochs, "nominal_epochs": nominal_epochs,
        "data": data.fingerprint, "learning_rate": learning_rate,
        "case_weight": case_weight, "case_positive_count": case_positive_count,
        "case_negative_count": case_negative_count, "source": "lal",
        "encoder_contract": serializable_query_encoder_contract("lal"),
        "case_sampler": "nearest-shared-label-v1/disjoint-hard64",
    }
    if not max_updates:
        contract["code_sha256"] = sha(Path(__file__))
    contract_hash = digest(contract)
    done = directory / "_SUCCESS.json"
    benchmark_path = directory / "BENCHMARK.json"
    if max_updates and benchmark_path.exists():
        value = read(benchmark_path)
        if value["contract"] != contract_hash or value["updates"] != max_updates:
            raise ValueError("Case benchmark contract changed")
        return value
    if done.exists():
        value = read(done)
        if value["contract"] != contract_hash:
            raise ValueError("Case-training resume contract changed")
        if any(sha(directory / name) != expected for name, expected in value["files"].items()):
            raise ValueError("Case-training completed checkpoint hash mismatch")
        return value
    seed_all()
    model = QueryEncoder(source="lal")
    bank = data.bank("lal", device="cuda")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate, weight_decay=.01,
    )
    steps = math.ceil(len(qids) / 16)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(.1 * steps * nominal_epochs)), steps * nominal_epochs,
    )
    state_path = directory / "resume.pt"
    start_epoch, start_pos, updates = 0, 0, 0
    if state_path.exists():
        receipt = state_path.with_suffix(".sha.json")
        if not receipt.exists() or read(receipt)["sha256"] != sha(state_path):
            raise ValueError("Case-training resume checkpoint lacks verified hash")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["contract"] != contract_hash:
            raise ValueError("Case-training checkpoint contract changed")
        model.load_state_dict(state["adapter"], strict=False)
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        set_rng(state["rng"])
        start_epoch, start_pos, updates = state["epoch"], state["position"], state["updates"]
    progress = Progress("query_case_" + directory.name, epochs * len(qids))
    begin = time.monotonic(); aggregate_doc = aggregate_case = aggregate_eligible = 0.0
    try:
        for epoch in range(start_epoch, epochs):
            order = list(qids); random.Random(112 + epoch).shuffle(order); model.train()
            for pos in range(start_pos if epoch == start_epoch else 0, len(order), 16):
                group = order[pos:pos + 16]; optimizer.zero_grad(set_to_none=True)
                update_doc = update_case = 0.0; eligible = 0
                for qid in group:
                    positive_cases, negative_cases = sampler.sample(
                        qid, epoch, positive_count=case_positive_count,
                        negative_count=case_negative_count,
                    )
                    case_ids = positive_cases + negative_cases
                    vectors = model([data.questions[qid]] + [data.questions[q] for q in case_ids])
                    query = vectors[0]
                    scores, indices = bank.mine(query)
                    scores = scores[0]; indices = indices[0]
                    ordering = torch.argsort(scores, descending=True, stable=True)
                    ranks = torch.argsort(ordering, stable=True) + 1
                    current = [data.doc_ids[i] for i in ordering.cpu().tolist()]
                    sources = store.rankings(qid)
                    negatives = select_negatives(current, data.gold[qid], sources, data.doc_ids, qid, epoch)
                    positives = sorted(data.gold[qid])
                    pi = torch.tensor([data.doc_row[d] for d in positives], device="cuda")
                    ni = torch.tensor([data.doc_row[d] for d in negatives], device="cuda")
                    positive_scores = bank.rescore(query, indices[pi])
                    negative_scores = bank.rescore(query, indices[ni])
                    doc_loss = multi_loss(positive_scores, negative_scores)
                    if epoch > 0:
                        doc_loss = doc_loss + .25 * boundary_loss(
                            positive_scores, negative_scores, ranks[pi], ranks[ni],
                        )
                    frozen = torch.as_tensor(data.query_vector(qid, "lal"), device="cuda")
                    loss = doc_loss + .05 * (1 - F.cosine_similarity(query[None], frozen[None]).mean())
                    case_value = torch.zeros((), device="cuda")
                    if positive_cases:
                        pc = vectors[1:1 + len(positive_cases)]
                        nc = vectors[1 + len(positive_cases):]
                        case_value = case_multi_loss(query, pc, nc)
                        loss = loss + case_weight * case_value
                        eligible += 1
                    scaled = loss / len(group)
                    if not torch.isfinite(scaled):
                        raise FloatingPointError("Non-finite dual document-case loss")
                    scaled.backward()
                    update_doc += float(doc_loss.detach()); update_case += float(case_value.detach())
                    del vectors, scores, indices, ordering, ranks
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                if float(gradient_norm) == 0.:
                    raise ValueError("Dual LAL query backbone received no nonzero gradients")
                optimizer.step(); scheduler.step(); updates += 1
                aggregate_doc += update_doc; aggregate_case += update_case; aggregate_eligible += eligible
                if updates % 16 == 0 or pos + len(group) == len(order):
                    checkpoint(state_path, model, optimizer, scheduler, contract=contract_hash,
                               epoch=epoch, position=pos + len(group), updates=updates)
                progress.update(
                    epoch * len(qids) + pos + len(group), doc_loss=update_doc / len(group),
                    case_loss=update_case / max(1, eligible), case_eligible=eligible,
                    gradient_norm=float(gradient_norm),
                )
                if max_updates and updates >= max_updates:
                    checkpoint(state_path, model, optimizer, scheduler, contract=contract_hash,
                               epoch=epoch, position=pos + len(group), updates=updates)
                    value = {
                        "contract": contract_hash, "benchmark_only": True, "updates": updates,
                        "seconds": time.monotonic() - begin,
                        "seconds_per_update": (time.monotonic() - begin) / updates,
                        "peak_vram": torch.cuda.max_memory_reserved(),
                        "case_eligible_fraction": aggregate_eligible / max(1, updates * 16),
                        "mean_doc_loss": aggregate_doc / max(1, updates * 16),
                        "mean_case_loss_on_eligible": aggregate_case / max(1, aggregate_eligible),
                    }
                    write(benchmark_path, value); return value
            checkpoint(directory / f"epoch-{epoch + 1}.pt", model, optimizer, scheduler,
                       contract=contract_hash, epoch=epoch + 1, position=0, updates=updates)
        result = {
            "contract": contract_hash, "training_qids": qids, "epochs": epochs,
            "updates": updates, "seconds": time.monotonic() - begin,
            "case_eligible_queries": len(sampler.eligible),
            "case_eligible_fraction": len(sampler.eligible) / len(qids),
            "mean_doc_loss": aggregate_doc / max(1, updates * 16),
            "mean_case_loss_on_eligible": aggregate_case / max(1, aggregate_eligible),
            "files": {p.name: sha(p) for p in directory.glob("epoch-*.pt")},
        }
        write(done, result); return result
    finally:
        del model, bank, optimizer, scheduler
        import gc
        gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def encode_query_vectors(data, qids, checkpoint_path, output, *, batch_size=8, source="lal"):
    """Encode a fold-scoped query set for adapted semantic case memory."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    vectors_path = output / "vectors.f32.npy"; ids_path = output / "query_ids.json"
    signature = digest([
        data.fingerprint, sha(checkpoint_path), source,
        serializable_query_encoder_contract(source), list(map(str, qids)),
    ])
    done = output / "_SUCCESS.json"
    if done.exists():
        value = read(done)
        if value["signature"] != signature or sha(vectors_path) != value["sha256"]:
            raise ValueError("Adapted query-vector cache mismatch")
        return value
    model = QueryEncoder(checkpoint=checkpoint_path, source=source); model.eval()
    temporary = output / "vectors.tmp.npy"
    matrix = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(qids), 1024),
    )
    progress = Progress("encode_case_" + output.name, len(qids))
    try:
        for start in range(0, len(qids), batch_size):
            ids = qids[start:start + batch_size]
            matrix[start:start + len(ids)] = model([data.questions[q] for q in ids]).cpu().numpy()
            progress.update(start + len(ids))
        matrix.flush(); del matrix
        os.replace(temporary, vectors_path)
        write(ids_path, list(map(str, qids)))
        value = {"signature": signature, "qids": len(qids), "sha256": sha(vectors_path)}
        write(done, value); return value
    finally:
        if "matrix" in locals():
            del matrix
        del model
        import gc
        gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def score_queries(data, qids, checkpoint_path, output, batch_size=4, source="e5"):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model = QueryEncoder(checkpoint=checkpoint_path, source=source); model.eval()
    bank = data.bank(source, device="cuda")
    signature = digest([data.fingerprint, sha(checkpoint_path) if checkpoint_path else "identity", source,
                        serializable_query_encoder_contract(source)])
    progress = Progress("score_" + output.name, len(qids))
    try:
        for start in range(0, len(qids), batch_size):
            ids = qids[start:start+batch_size]
            pending = [q for q in ids if not (output / f"{q}.json").exists()]
            for q in set(ids)-set(pending):
                row = read(output / f"{q}.json")
                if row["signature"] != signature or row.get('payload_hash') != digest([row['order'],row['scores']]):
                    raise ValueError("Scoring resume mismatch")
            if pending:
                vectors = model([data.questions[q] for q in pending])
                scores, _ = bank.mine(vectors)
                order = torch.argsort(scores, dim=1, descending=True, stable=True)
                for q, s, indices, v in zip(pending, scores.cpu().numpy(), order.cpu().numpy(), vectors.cpu().numpy()):
                    row = {"signature": signature, "order": [data.doc_ids[i] for i in indices], "scores": s[indices].tolist(),
                           "frozen_query_cosine": float(v @ data.query_vector(q, source))}
                    row['payload_hash'] = digest([row['order'],row['scores']])
                    write(output / f"{q}.json", row)
            progress.update(start+len(ids))
        write(output / "_SUCCESS.json", {"signature": signature, "qids": qids})
    finally:
        del model, bank
        import gc
        gc.collect(); torch.cuda.empty_cache()
