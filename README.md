# Sift

![CI](https://github.com/HussainQadri/sift/actions/workflows/ci.yml/badge.svg)

Sift is local semantic code search for codebases. Ask in natural language and jump straight to the functions that matter.

<p align="center">
  <img src="assets/sift-demo.gif" alt="Sift CLI demo">
</p>

It uses Tree-sitter to extract functions, embeds their full source with a code-oriented model, stores the index locally, and ranks matches with cosine similarity.

## Usage

Ingest supported source files from a directory:

```bash
cargo run -- ingest ./src
```

Search the stored index separately:

```bash
cargo run -- "load saved index records"
```

After installing the binary:

```bash
sift ingest ./src
sift "extract functions from a syntax tree"
```

## Current Pipeline

Ingestion:

```text
directory
-> source files
-> Tree-sitter function extraction
-> Jina code embeddings
-> .sift/index.json
```

Search:

```text
query
-> Jina query embedding
-> load .sift/index.json
-> cosine similarity against saved function embeddings
-> print top matches with source locations
```


Full function source is embedded for retrieval. Results remain compact by
printing the function header with its file path, line number, and similarity
score.

## Supported Files

- Rust: `.rs`
- Python: `.py`
- C++: `.cpp`

## Current Limitations

- The embedding model is currently initialized in the per-file indexing path.
- Search performs an exact cosine scan over all saved vectors.
- `.sift/index.json` is inspectable prototype storage, not the intended format
  for very large indexes.
- Ranking is semantic-only; exact identifier-aware ranking is not implemented.

## Direction

Exact cosine search is the correctness baseline. The next substantial indexing
work is a custom HNSW approximate nearest-neighbour implementation for large
codebases, evaluated against exhaustive search with latency and recall
benchmarks.

Planned work:

- Initialize the embedding model once per ingestion run and batch embeddings.
- Add ranking/evaluation queries and identifier-aware retrieval signals.
- Implement in-memory HNSW search and compare it against exact cosine search.
- Replace JSON vector/graph storage with a compact persistent representation.
