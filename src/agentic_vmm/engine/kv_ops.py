# src/agentic_vmm/engine/kv_ops.py
"""
Defensive KV-cache sequence branching. Every call validates seq ids against
the C-layer n_seq_max BEFORE touching libllama.so, since llama_memory_seq_cp
has no bounds checking internally and will segfault on out-of-range ids.
"""

from __future__ import annotations

import logging

from llama_cpp import llama_cpp as llama_cpp_lib

logger = logging.getLogger(__name__)


class KVSequenceError(Exception):
    pass


def _n_seq_max(engine) -> int:
    n = engine.n_seq_max()
    if n <= 1:
        raise KVSequenceError(
            f"engine reports n_seq_max={n}; branching unsupported on this "
            "context (check LlamaEngine init, not this call)."
        )
    return n


def seq_copy(
    engine,
    src_seq_id: int,
    dst_seq_id: int,
    p0: int = -1,
    p1: int = -1,
) -> None:
    """
    Safe wrapper around llama_memory_seq_cp(mem, src, dst, p0, p1).

    p0 == -1 and p1 == -1 mean "whole sequence" per llama.cpp convention.
    """
    n_seq_max = _n_seq_max(engine)

    if not (0 <= src_seq_id < n_seq_max):
        raise KVSequenceError(
            f"src_seq_id={src_seq_id} out of range [0, {n_seq_max})"
        )
    if not (0 <= dst_seq_id < n_seq_max):
        raise KVSequenceError(
            f"dst_seq_id={dst_seq_id} out of range [0, {n_seq_max}). "
            f"Requested branch target exceeds allocated C-layer sequence slots."
        )
    if src_seq_id == dst_seq_id:
        raise KVSequenceError("src_seq_id == dst_seq_id, no-op copy rejected")
    if p0 != -1 and p1 != -1 and p0 > p1:
        raise KVSequenceError(f"invalid range p0={p0} > p1={p1}")

    mem = getattr(engine, "mem", None)
    if mem is None:
        raise KVSequenceError(
            "engine.mem is None; llama_get_memory(ctx) was never bound "
            "(engine not fully initialized)."
        )

    logger.debug(
        "seq_copy: src=%d dst=%d p0=%d p1=%d (n_seq_max=%d)",
        src_seq_id, dst_seq_id, p0, p1, n_seq_max,
    )

    llama_cpp_lib.llama_memory_seq_cp(mem, src_seq_id, dst_seq_id, p0, p1)


def seq_rm(engine, seq_id: int, p0: int = -1, p1: int = -1) -> None:
    n_seq_max = _n_seq_max(engine)
    if not (0 <= seq_id < n_seq_max):
        raise KVSequenceError(f"seq_id={seq_id} out of range [0, {n_seq_max})")
    llama_cpp_lib.llama_memory_seq_rm(engine.mem, seq_id, p0, p1)


def seq_keep(engine, seq_id: int) -> None:
    n_seq_max = _n_seq_max(engine)
    if not (0 <= seq_id < n_seq_max):
        raise KVSequenceError(f"seq_id={seq_id} out of range [0, {n_seq_max})")
    llama_cpp_lib.llama_memory_seq_keep(engine.mem, seq_id)
