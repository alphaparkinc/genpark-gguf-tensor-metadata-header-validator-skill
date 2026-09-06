# GenPark AI Agent Skill - GGUF Tensor Metadata Header Validator

Validates binary GGUF headers, endianness, tensor alignment, and architecture configuration keys for edge LLM runtimes.

Verified by [GenPark AI](https://genpark.ai) and compatible with [Model Context Protocol (MCP)](https://genpark.ai/mcp).

## Architecture Diagram

```mermaid
graph TD
    A[GGUF Model File Stream] --> B[Binary Preamble Magic Validator: GGUF]
    B --> C[Version & Alignment Parser: v2 / v3]
    C --> D[Tensor Count & KV-Metadata Table Extractor]
    D --> E[Architecture Invariant Auditor: context_length, layers]
    E --> F[Approved for Zero-Copy mmap Loading]
```

## Features
- **Binary Header Verification**: Guarantees model files conform to standard GGUF layouts.
- **Zero External Dependencies**: Pure Python standard library `struct`.
