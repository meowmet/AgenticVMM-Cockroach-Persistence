# src/agentic_vmm/branch/tree.py
"""
BranchTree: logical parent-child graph of BranchNode commits.

Pure bookkeeping — no C-level / llama.cpp calls here. BranchManager owns
the wiring between this tree and the physical KV-cache sequences.
"""

from __future__ import annotations

from typing import Iterator, Optional


class BranchTreeError(Exception):
    pass


class BranchTree:
    def __init__(self) -> None:
        self._nodes: dict[str, "BranchNode"] = {}
        self._children: dict[str, list[str]] = {}
        self._root_id: Optional[str] = None

    # -- mutation -----------------------------------------------------

    def add_node(self, node: "BranchNode") -> "BranchNode":
        if node.node_id in self._nodes:
            raise BranchTreeError(f"duplicate node_id: {node.node_id}")

        if node.parent_id is None:
            if self._root_id is not None:
                raise BranchTreeError(
                    f"tree already has a root ({self._root_id}); "
                    "cannot add a second root node"
                )
            self._root_id = node.node_id
        elif node.parent_id not in self._nodes:
            raise BranchTreeError(f"unknown parent_id: {node.parent_id}")

        self._nodes[node.node_id] = node
        self._children.setdefault(node.node_id, [])
        if node.parent_id is not None:
            self._children[node.parent_id].append(node.node_id)
        return node

    def remove_subtree(self, node_id: str) -> list["BranchNode"]:
        """Remove node_id and all its descendants. Returns removed nodes."""
        self._require(node_id)
        removed: list[BranchNode] = []
        stack = [node_id]
        while stack:
            current = stack.pop()
            stack.extend(self._children.get(current, []))
            removed.append(self._nodes.pop(current))
            self._children.pop(current, None)

        parent_id = removed[-1].parent_id if removed else None
        # removed[-1] is the originally requested node (DFS order pops it last
        # only if it has no children; safer to recompute directly)
        node = next((n for n in removed if n.node_id == node_id), None)
        parent_id = node.parent_id if node else None
        if parent_id is not None and parent_id in self._children:
            self._children[parent_id] = [
                c for c in self._children[parent_id] if c != node_id
            ]
        if node_id == self._root_id:
            self._root_id = None
        return removed

    # -- lookups --------------------------------------------------------

    def get(self, node_id: str) -> "BranchNode":
        return self._require(node_id)

    def children_of(self, node_id: str) -> list["BranchNode"]:
        self._require(node_id)
        return [self._nodes[cid] for cid in self._children.get(node_id, [])]

    def root(self) -> "BranchNode":
        if self._root_id is None:
            raise BranchTreeError("tree is empty; no root node")
        return self._nodes[self._root_id]

    def is_empty(self) -> bool:
        return self._root_id is None

    def __iter__(self) -> Iterator["BranchNode"]:
        return iter(self._nodes.values())

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_history(self, node_id: str) -> list["BranchNode"]:
        """Root-to-node_id path, root first."""
        self._require(node_id)
        path: list[BranchNode] = []
        current: Optional[str] = node_id
        while current is not None:
            node = self._nodes[current]
            path.append(node)
            current = node.parent_id
        path.reverse()
        return path

    # -- rendering --------------------------------------------------------

    def render_ascii(self, active_node_id: Optional[str] = None) -> str:
        """Git-log-graph-style ASCII rendering of the whole tree."""
        if self.is_empty():
            return "(empty tree)"

        lines: list[str] = []
        self._render_node(self._root_id, "", True, active_node_id, lines)
        return "\n".join(lines)

    def _render_node(
        self,
        node_id: str,
        prefix: str,
        is_last: bool,
        active_node_id: Optional[str],
        lines: list[str],
    ) -> None:
        node = self._nodes[node_id]
        connector = "└─" if is_last else "├─"
        marker = " [HEAD]" if node_id == active_node_id else ""
        label = self._label(node)
        branch_glyph = "┬" if len(self._children.get(node_id, [])) > 1 else "─"

        if prefix == "":
            lines.append(f"●{branch_glyph}─ {label}{marker}")
        else:
            lines.append(f"{prefix}{connector}{branch_glyph}─ {label}{marker}")

        children = self._children.get(node_id, [])
        child_prefix = prefix + ("   " if is_last else "│  ")
        for i, child_id in enumerate(children):
            self._render_node(
                child_id,
                child_prefix,
                i == len(children) - 1,
                active_node_id,
                lines,
            )

    @staticmethod
    def _label(node: "BranchNode") -> str:
        short_id = node.node_id[:8]
        preview = (node.prompt_text or node.generated_text or "").strip()
        preview = preview.replace("\n", " ")
        if len(preview) > 40:
            preview = preview[:37] + "..."
        return f"{short_id} (seq={node.seq_id}) {preview}".rstrip()

    # -- internals --------------------------------------------------------

    def _require(self, node_id: str) -> "BranchNode":
        node = self._nodes.get(node_id)
        if node is None:
            raise BranchTreeError(f"unknown node_id: {node_id}")
        return node


from agentic_vmm.branch.node import BranchNode  # noqa: E402  (avoid circular import at module load)
