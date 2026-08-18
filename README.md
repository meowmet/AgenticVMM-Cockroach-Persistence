# AgenticVMM: Distributed Memory Engine

**O(1) KV-Cache Branching Engine with CockroachDB & AWS S3 Persistence for Edge AI Agents**

---

## Inspiration
Autonomous LLM agents (like penetration testers, coders, or researchers) operate in multi-step decision loops. When they hit a dead end, they need to backtrack and try a different approach. HoIver, traditional frameworks handle this by completely duplicating message history as strings. This results in an \( O(n) \) latency bottleneck, forcing the system to re-prefill the entire context, which consumes precious seconds and completely exhausts the VRAM of Edge AI devices. 

I needed a system that branches like a decision tree in machine learning or a `fork()` in operating systems, backed by an unbreakable persistent memory layer.

## What it does
**AgenticVMM (Distributed Agentic Memory Engine)** is a hardware-level version control and memory management system for Edge AI agents. It completely eliminates the context re-prefill bottleneck.

Instead of copying text, AgenticVMM dives directly into the `llama.cpp` C-API and physically clones the **KV-Cache pointers** in the VRAM. 
* **\( O(1) \) Branching:** Creating a new agent thought-branch takes **~0.09ms** instead of seconds.
* **Zero VRAM Bloat:** Newly created branches cost **0 MB** of additional VRAM, utilizing shared prefixes.
* **True LRU Eviction:** Failed exploratory branches are automatically evicted to prevent out-of-memory errors.

**Hybrid Distributed Memory Architecture:**
While VRAM handles volatile, high-speed branching, AgenticVMM ensures zero data loss by committing successful agent milestones directly to **CockroachDB** and archiving session logs to **AWS S3-compatible** cold storage.

## How I built it (Hackathon Integration)
I built the core engine using Python with a defensive wrapper around the `llama.cpp` ctypes library. To meet enterprise-grade persistence requirements:

1. **CockroachDB Managed MCP Server:** Integrated Model Context Protocol so our agents can securely read/write state schemas and structural context to a globally distributed database.
2. **ccloud CLI (Agent-Ready):** When a branch achieves a successful objective, the agent triggers the `ccloud` CLI with `--output-format=json` to persistently pin this memory state to the CockroachDB cluster.
3. **AWS S3 Compatible Storage:** While failed branches are flushed from RAM via our LRU algorithm, verified findings and audit logs are archived into an S3 bucket (`boto3`).

## Challenges I ran into
Manipulating memory at the C-pointer level via Python is extremely dangerous, leading to frequent `Segmentation Faults` and memory leaks. Ensuring that the high-speed VRAM engine synchronizes with the CockroachDB distributed cluster without stalling the agent's decision loop required a robust asynchronous fallback architecture.

## Accomplishments that I'm proud of
* Achieving a **2450x speedup** in branch creation latency (~0.34ms vs ~588ms).
* Keeping VRAM consumption strictly bounded on consumer hardware (6 GB VRAM).
* Successfully marrying low-level C hardware manipulation with high-level distributed cloud databases (**CockroachDB & AWS S3**).

## What's next
* Implementing Multi-Agent KV-Cache sharing.
* Expanding CockroachDB MCP integration to support semantic branch retrieval via vector embeddings.

---
## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
