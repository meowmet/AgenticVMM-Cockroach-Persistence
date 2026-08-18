# src/agentic_vmm/branch/node.py
"""
BranchNode: a single commit in the conversation branch tree.

Each node binds a logical conversation turn to a physical llama.cpp
KV-cache sequence id (seq_id). The node itself is immutable data;
tree/graph logic lives in BranchTree, C-level KV operations live in
BranchManager.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _new_node_id() -> str:
    return str(uuid.uuid4())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BranchNode:
    """A single commit (conversation turn) in the branch tree."""

    seq_id: int
    prompt_text: str = ""
    generated_text: str = ""
    kv_pos: int = 0
    parent_id: Optional[str] = None
    is_pinned: bool = False
    last_accessed_at: float = field(default_factory=time.time)
    node_id: str = field(default_factory=_new_node_id)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_root(self) -> bool:
        return self.parent_id is None

    def turn_text(self) -> str:
        """Render this node's contribution to the conversation transcript."""
        parts = []
        if self.prompt_text:
            parts.append(f"User: {self.prompt_text}")
        if self.generated_text:
            parts.append(f"Assistant: {self.generated_text}")
        return "\n".join(parts)

    def __repr__(self) -> str:
        short_id = self.node_id[:8]
        parent_short = self.parent_id[:8] if self.parent_id else "None"
        return (
            f"BranchNode(id={short_id}, parent={parent_short}, "
            f"seq_id={self.seq_id})"
        )
