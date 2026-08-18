# src/agentic_vmm/engine/llama_engine.py
"""
Root-cause fix: llama-cpp-python's high-level Llama() wrapper drops n_seq_max
even when passed as a kwarg (silently ignored in ctor -> llama_context_params
in several llama-cpp-python versions). Fix: bypass Llama() for context
creation and build model + context via low-level llama_cpp bindings directly,
setting n_seq_max on the actual llama_context_params struct before
llama_new_context_with_model().
"""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path

import llama_cpp
from llama_cpp import llama_cpp as llama_cpp_lib  # low-level cffi/ctypes bindings

logger = logging.getLogger(__name__)


class LlamaEngine:
    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 8192,
        n_seq_max: int = 8,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        n_batch: int = 2048,
        n_ubatch: int = 512,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.model_path = str(model_path)
        self.n_ctx = n_ctx
        self.n_seq_max_requested = n_seq_max
        self.verbose = verbose

        llama_cpp_lib.llama_backend_init()

        # --- model params (low-level) ---
        model_params = llama_cpp_lib.llama_model_default_params()
        model_params.n_gpu_layers = n_gpu_layers

        self.model = llama_cpp_lib.llama_load_model_from_file(
            self.model_path.encode("utf-8"), model_params
        )
        if self.model is None:
            raise RuntimeError(f"llama_load_model_from_file failed: {self.model_path}")

        # --- context params (low-level, THIS is where n_seq_max actually lands) ---
        ctx_params = llama_cpp_lib.llama_context_default_params()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = n_batch
        ctx_params.n_ubatch = n_ubatch
        ctx_params.n_seq_max = n_seq_max  # <-- the fix
        if n_threads is not None:
            ctx_params.n_threads = n_threads
            ctx_params.n_threads_batch = n_threads

        self.ctx = llama_cpp_lib.llama_new_context_with_model(self.model, ctx_params)
        if self.ctx is None:
            llama_cpp_lib.llama_free_model(self.model)
            raise RuntimeError("llama_new_context_with_model failed")

        # memory handle used by KV branching ops (new API: llama_get_memory)
        self.mem = llama_cpp_lib.llama_get_memory(self.ctx)

        self._verify_n_seq_max()

    def _verify_n_seq_max(self) -> None:
        actual = self.n_seq_max()
        if actual < self.n_seq_max_requested:
            self.close()
            raise RuntimeError(
                f"n_seq_max mismatch after low-level init: requested="
                f"{self.n_seq_max_requested}, actual(C)={actual}. "
                "Model/context did not accept requested value."
            )
        logger.info("LlamaEngine: n_seq_max verified at C layer = %d", actual)

    def n_seq_max(self) -> int:
        """Query actual n_seq_max from the live context (source of truth)."""
        return llama_cpp_lib.llama_n_seq_max(self.ctx)

    def n_ctx_train(self) -> int:
        return llama_cpp_lib.llama_n_ctx_train(self.model)

    def close(self) -> None:
        if getattr(self, "ctx", None):
            llama_cpp_lib.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "model", None):
            llama_cpp_lib.llama_free_model(self.model)
            self.model = None

    # -- inference (tokenize / decode / sample) --------------------------

    def tokenize(self, text: str, add_bos: bool = True) -> list[int]:
        n_max = len(text.encode("utf-8")) + 16
        buf = (llama_cpp_lib.llama_token * n_max)()
        vocab = llama_cpp_lib.llama_model_get_vocab(self.model)
        n = llama_cpp_lib.llama_tokenize(
            vocab,
            text.encode("utf-8"),
            len(text.encode("utf-8")),
            buf,
            n_max,
            add_bos,
            True,  # parse_special
        )
        if n < 0:
            raise RuntimeError(f"tokenize buffer too small, need {-n}")
        return list(buf[:n])

    def token_to_piece(self, token_id: int) -> str:
        buf = ctypes.create_string_buffer(64)
        vocab = llama_cpp_lib.llama_model_get_vocab(self.model)
        n = llama_cpp_lib.llama_token_to_piece(
            vocab, token_id, buf, len(buf), 0, True
        )
        if n < 0:
            buf = ctypes.create_string_buffer(-n)
            llama_cpp_lib.llama_token_to_piece(
                vocab, token_id, buf, len(buf), 0, True
            )
        return buf.value.decode("utf-8", errors="ignore")

    def decode_tokens(
        self,
        seq_id: int,
        tokens: list[int],
        pos_start: int,
    ) -> None:
        """Feed `tokens` into the KV-cache for `seq_id`, starting at pos_start."""
        n = len(tokens)
        if n == 0:
            return
            
        chunk_size = 512
        for offset in range(0, n, chunk_size):
            chunk = tokens[offset:offset + chunk_size]
            cn = len(chunk)
            batch = llama_cpp_lib.llama_batch_init(cn, 0, 1)
            try:
                for i, tok in enumerate(chunk):
                    batch.token[i] = tok
                    batch.pos[i] = pos_start + offset + i
                    batch.n_seq_id[i] = 1
                    batch.seq_id[i][0] = seq_id
                    batch.logits[i] = 1 if (offset + i == n - 1) else 0
                batch.n_tokens = cn

                ret = llama_cpp_lib.llama_decode(self.ctx, batch)
                if ret != 0:
                    raise RuntimeError(f"llama_decode failed (seq_id={seq_id}, chunk offset={offset}), ret={ret}")
            finally:
                llama_cpp_lib.llama_batch_free(batch)

    def sample_greedy(self) -> int:
        """Greedy-sample the next token from the logits of the last decode call."""
        n_vocab = llama_cpp_lib.llama_vocab_n_tokens(
            llama_cpp_lib.llama_model_get_vocab(self.model)
        )
        logits_ptr = llama_cpp_lib.llama_get_logits_ith(self.ctx, -1)
        logits = ctypes.cast(logits_ptr, ctypes.POINTER(ctypes.c_float * n_vocab)).contents
        best_id = max(range(n_vocab), key=lambda i: logits[i])
        return best_id

    def generate(
        self,
        seq_id: int,
        prompt_tokens: list[int],
        pos_start: int,
        max_new_tokens: int = 64,
        eos_token_ids: set[int] | None = None,
    ) -> tuple[list[int], str]:
        """
        Decode prompt_tokens into seq_id starting at pos_start, then greedily
        sample max_new_tokens more tokens on that same seq_id.
        Returns (generated_token_ids, generated_text).
        """
        if eos_token_ids is None:
            eos_token_ids = {llama_cpp_lib.llama_vocab_eos(
                llama_cpp_lib.llama_model_get_vocab(self.model)
            )}

        self.decode_tokens(seq_id, prompt_tokens, pos_start)
        pos = pos_start + len(prompt_tokens)

        generated_ids: list[int] = []
        text_parts: list[str] = []

        for _ in range(max_new_tokens):
            next_id = self.sample_greedy()
            if next_id in eos_token_ids:
                break
            generated_ids.append(next_id)
            text_parts.append(self.token_to_piece(next_id))
            self.decode_tokens(seq_id, [next_id], pos)
            pos += 1

        return generated_ids, "".join(text_parts)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
