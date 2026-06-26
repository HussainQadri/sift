# Sift

![CI](https://github.com/HussainQadri/sift/actions/workflows/ci.yml/badge.svg)

Sift is local semantic code search for codebases. Ask in natural language and jump straight to the functions that matter.

<p align="center">
  <img src="assets/sift-demo.gif" alt="Sift CLI demo">
</p>

It uses Tree-sitter to extract functions, embeds their full source with a code-oriented model, stores the index locally, and searches it with an in-memory HNSW index while keeping exact cosine search around as a baseline.

## Install

```bash
cargo install --path .
```

## Usage

Ingest supported source files from a directory:

```bash
sift ingest ./src
```

Search the stored index separately:

```bash
sift "load saved index records"
```

## Supported Files

- Rust: `.rs`
- Python: `.py`
- C++: `.cpp`

## How It Works

Ingestion:

```text
directory
-> source files
-> Tree-sitter function extraction
-> length-aware batches of Jina code embeddings
-> .sift-index/index.json
```

Search:

```text
query
-> Jina query embedding
-> load .sift-index/index.json
-> rebuild an in-memory HNSW index from saved embeddings
-> compare HNSW results with exact cosine results while HNSW is being tuned
-> print top matches with source locations
```

Full function source is embedded for retrieval. Results remain compact by
printing the function header with its file path, line number, and similarity
score.

## Roadmap

Current search rebuilds an in-memory HNSW graph from the saved JSON index and
prints exact cosine results alongside it while recall is being checked.

Planned work:

- Add ranking/evaluation queries and identifier-aware retrieval signals.
- Tune HNSW parameters and measure recall against exact cosine search.
- Replace JSON vector/graph storage with a compact persistent representation.
