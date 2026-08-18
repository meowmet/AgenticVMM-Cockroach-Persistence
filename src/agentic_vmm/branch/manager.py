# src/agentic_vmm/branch/manager.py
"""
BranchManager: bridges the logical BranchTree (conversation commits) with
the physical llama.cpp KV-cache sequences owned by LlamaEngine.

Sequence-id allocation policy:
  - seq_id 0 is reserved for the initial/root branch.
  - Additional seq_ids (1..n_seq_max-1) are handed out from a free pool
    when create_branch() is called, and returned to the pool when a
    branch subtree is dropped.
  - checkout() never allocates; it only points the "active" pointer at
    an existing node and its already-materialized seq_id.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from agentic_vmm.branch.node import BranchNode
from agentic_vmm.branch.tree import BranchTree, BranchTreeError
from agentic_vmm.engine.kv_ops import KVSequenceError, seq_copy, seq_rm

logger = logging.getLogger(__name__)

ROOT_SEQ_ID = 0


class BranchManagerError(Exception):
    pass


class BranchManager:
    def __init__(self, engine) -> None:
        """
        engine: an initialized LlamaEngine instance (already exposes
                n_seq_max(), mem, ctx, etc. — see llama_engine.py).
        """
        self.engine = engine
        self.tree = BranchTree()

        n_seq_max = self.engine.n_seq_max()
        if n_seq_max < 1:
            raise BranchManagerError(f"invalid engine.n_seq_max()={n_seq_max}")

        self._free_seq_ids: list[int] = list(range(1, n_seq_max))
        self._n_seq_max = n_seq_max

        root = BranchNode(seq_id=ROOT_SEQ_ID, parent_id=None)
        self.tree.add_node(root)
        self._active_node_id: str = root.node_id
        self._seq_owner: dict[int, str] = {ROOT_SEQ_ID: root.node_id}

        logger.info(
            "BranchManager initialized: n_seq_max=%d, root=%s",
            n_seq_max, root.node_id,
        )

    # -- public API -----------------------------------------------------

    @property
    def active_node_id(self) -> str:
        return self._active_node_id

    def active_node(self) -> BranchNode:
        return self.tree.get(self._active_node_id)

    def commit_and_generate(self, user_input: str, max_new_tokens: int = 40) -> BranchNode:
        """
        Tokenize user_input (as delta), append to current branch, generate response,
        and update kv_pos.
        """
        parent = self.active_node()
        prompt = f"\nUser: {user_input}\nAssistant:"
        
        # Sadece delta, add_bos=False
        tokens = self.engine.tokenize(prompt, add_bos=False)
        
        gen_ids, gen_text = self.engine.generate(
            seq_id=parent.seq_id,
            prompt_tokens=tokens,
            pos_start=parent.kv_pos,
            max_new_tokens=max_new_tokens
        )
        
        new_kv_pos = parent.kv_pos + len(tokens) + len(gen_ids)
        
        node = BranchNode(
            seq_id=parent.seq_id,
            parent_id=parent.node_id,
            prompt_text=user_input,
            generated_text=gen_text,
            kv_pos=new_kv_pos
        )
        self.tree.add_node(node)
        self._seq_owner[node.seq_id] = node.node_id
        self._active_node_id = node.node_id
        logger.debug("commit_and_generate: %s -> %s (seq=%d)", parent.node_id, node.node_id, node.seq_id)
        return node

    def create_branch(self, from_node_id: str) -> BranchNode:
        """
        Fork a new branch from an existing commit. Physically copies the
        KV-cache of from_node's seq_id into a freshly allocated seq_id via
        the C-level seq_copy (already guarded against out-of-range ids).
        """
        source_node = self.tree.get(from_node_id)

        # Slot doluysa otomatik LRU eviction dene
        self._evict_lru_if_needed()

        if not self._free_seq_ids:
            raise BranchManagerError(
                f"no free KV sequence slots available "
                f"(n_seq_max={self._n_seq_max} exhausted); "
                "drop an existing branch before creating a new one"
            )

        new_seq_id = self._free_seq_ids.pop(0)
        try:
            seq_copy(self.engine, src_seq_id=source_node.seq_id, dst_seq_id=new_seq_id)
        except KVSequenceError:
            self._free_seq_ids.insert(0, new_seq_id)  # give the slot back
            raise

        branch_node = BranchNode(
            seq_id=new_seq_id,
            parent_id=source_node.node_id,
            kv_pos=source_node.kv_pos,
            metadata={"branch_point": True},
        )
        self.tree.add_node(branch_node)
        self._seq_owner[new_seq_id] = branch_node.node_id
        self._active_node_id = branch_node.node_id

        logger.info(
            "create_branch: forked seq=%d from node=%s -> new node=%s (seq=%d)",
            source_node.seq_id, from_node_id, branch_node.node_id, new_seq_id,
        )
        return branch_node

    def checkout(self, node_id: str) -> BranchNode:
        """Switch the active pointer to an existing commit / branch tip."""
        node = self.tree.get(node_id)
        node.last_accessed_at = time.time()
        self._active_node_id = node.node_id
        logger.debug("checkout: active_node -> %s (seq=%d)", node.node_id, node.seq_id)
        return node

    def pin(self, node_id: str) -> None:
        node = self.tree.get(node_id)
        node.is_pinned = True
        node.last_accessed_at = time.time()
        logger.debug("PINNED: %s (seq=%d)", node.node_id, node.seq_id)

    def unpin(self, node_id: str) -> None:
        node = self.tree.get(node_id)
        if node.seq_id == ROOT_SEQ_ID:
            raise BranchManagerError("Root node (seq=0) is critical context, cannot be unpinned!")
        node.is_pinned = False
        node.last_accessed_at = time.time()
        logger.debug("UNPINNED: %s (seq=%d)", node.node_id, node.seq_id)

    def reset_hard(self, target_node_id: str) -> BranchNode:
        """
        Yerinde geri sarma (git reset --hard). 
        WITHOUT ALLOCATING a new seq_id, post-kv_pos on the current sequence 
        deletes all KV tensors at C level (seq_rm).
        """
        target_node = self.tree.get(target_node_id)
        current_node = self.active_node()

        if target_node.seq_id != current_node.seq_id:
            raise BranchManagerError("hard_reset can only be done to ancestor nodes on the same sequence. Use checkout() for different branches.")

        # Deleting all tokens after target_node.kv_pos at C level
        cutoff_pos = target_node.kv_pos
        seq_rm(self.engine, seq_id=target_node.seq_id, p0=cutoff_pos, p1=-1)

        # Clean up pruned subbranches in the tree
        children = self.tree.children_of(target_node_id)
        for child in children:
            self.tree.remove_subtree(child.node_id)

        self._active_node_id = target_node.node_id
        logger.info("reset_hard: seq=%d, %d KV cleared after pos -> node=%s", 
                    target_node.seq_id, cutoff_pos, target_node.node_id)
        return target_node

    def drop(self, node_id: str) -> None:
        """
        Remove a branch subtree and free its associated KV sequence(s).
        Cannot drop the root (seq_id 0) or the currently active node.
        """
        node = self.tree.get(node_id)
        if node.node_id == self.tree.root().node_id:
            raise BranchManagerError("cannot drop the root node")
        if node.node_id == self._active_node_id:
            raise BranchManagerError(
                "cannot drop the currently active node; checkout elsewhere first"
            )

        removed = self.tree.remove_subtree(node_id)
        freed_seq_ids = {n.seq_id for n in removed if n.seq_id != ROOT_SEQ_ID}

        for seq_id in freed_seq_ids:
            still_referenced = any(
                other.seq_id == seq_id for other in self.tree
            )
            if still_referenced:
                continue
            try:
                seq_rm(self.engine, seq_id)
            except KVSequenceError as exc:
                logger.warning("seq_rm(%d) failed during drop: %s", seq_id, exc)
            self._seq_owner.pop(seq_id, None)
            if seq_id not in self._free_seq_ids:
                self._free_seq_ids.append(seq_id)

        logger.info(
            "drop: removed %d node(s) under %s, freed seq_ids=%s",
            len(removed), node_id, sorted(freed_seq_ids),
        )

    def get_active_context(self) -> str:
        return self._render_context(self._active_node_id)

    def get_context(self, node_id: str) -> str:
        return self._render_context(node_id)

    def render_tree(self) -> str:
        return self.tree.render_ascii(active_node_id=self._active_node_id)

    def slot_status(self) -> dict:
        """VRAM sequence slot doluluk durumu."""
        used_seq_ids = sorted({n.seq_id for n in self.tree})
        return {
            "n_seq_max": self._n_seq_max,
            "free_slots": len(self._free_seq_ids),
            "used_slots": self._n_seq_max - len(self._free_seq_ids),
            "free_seq_ids": sorted(self._free_seq_ids),
            "used_seq_ids": used_seq_ids,
            "active_seq_id": self.active_node().seq_id,
            "active_kv_pos": self.active_node().kv_pos,
            "total_nodes": len(self.tree),
        }

    # -- internals -----------------------------------------------------

    def _evict_lru_if_needed(self) -> None:
        """Automatically clears C-level slot of the oldest passive branch when no empty KV slot is left."""
        if self._free_seq_ids:
            return

        # Scan the oldest accessed nodes, excluding root and active node
        active_seq = self.active_node().seq_id
        candidates = [
            node for node in self.tree
            if node.seq_id != ROOT_SEQ_ID
            and node.seq_id != active_seq
            and not node.is_pinned
        ]

        if not candidates:
            raise BranchManagerError("All slots are pinned or active. LRU eviction failed!")

        # TRUE LRU: Find the oldest "accessed" node
        lru_node = min(candidates, key=lambda n: n.last_accessed_at)
        logger.info("LRU Eviction (Time: %f): seq=%d temizleniyor.", lru_node.last_accessed_at, lru_node.seq_id)
        self.drop(lru_node.node_id)

    def _render_context(self, node_id: str) -> str:
        try:
            history = self.tree.get_history(node_id)
        except BranchTreeError as exc:
            raise BranchManagerError(str(exc)) from exc

        turns = [n.turn_text() for n in history if n.turn_text()]
        return "\n\n".join(turns)
