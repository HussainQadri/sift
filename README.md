# Sift

![CI](https://github.com/HussainQadri/sift/actions/workflows/ci.yml/badge.svg)

Sift is local semantic code search for codebases. Ask in natural language and
jump straight to the functions that matter.

<p align="center">
  <img src="assets/sift-demo.gif" alt="Sift CLI demo">
</p>

Sift uses Tree-sitter to extract functions and methods, embeds their full source
with Jina Embeddings v2 Base Code, and stores the resulting index locally. Exact
cosine search is the default; the custom persisted HNSW index provides an
optional approximate search mode.

Embedding inference and indexing run locally. On first use, FastEmbed downloads
the model into the operating system's cache directory.

## Install

Sift requires a current stable Rust toolchain. From this repository, run:

```bash
cargo install --path .
```

## Usage

Run Sift from the directory in which you want it to store `.sift-index`. To
index the current codebase:

```bash
cd path/to/codebase
sift ingest .
```

If no path is supplied, `ingest` defaults to the current directory:

```bash
sift ingest
```

Ingestion walks the directory recursively, follows the standard ignore rules
used by the `ignore` crate (including `.gitignore`), and replaces the existing
local index.

Search with exact cosine similarity:

```bash
sift "load saved index records"
```

Return a different number of results (the default is 3):

```bash
sift --top 5 "load saved index records"
```

Use the persisted HNSW graph for approximate search:

```bash
sift --hnsw --top 5 "load saved index records"
```

Each result includes its cosine similarity score, source path, one-based line
number, and syntax-highlighted function or method header.

## Benchmarking HNSW

After ingesting a codebase, create a text file with one query per line. Blank
lines are ignored:

```text
load the stored index
extract functions from source
calculate vector similarity
```

Then compare HNSW with exact search:

```bash
sift benchmark --queries queries.txt --top 10 --runs 50
```

The benchmark reports average recall across the queries and median search time
in milliseconds for both implementations. Query embedding time is not included.
`--top` defaults to 10 and `--runs` defaults to 50.

## Supported Files

- Rust: `.rs`
- Python: `.py`
- C++: `.cpp`
- Java: `.java`

Sift currently indexes function definitions in Rust, Python, and C++, and method
declarations in Java.

## Index Files

Each ingestion writes two files relative to the directory where Sift is run:

```text
.sift-index/
├── index.json  # function metadata, source, and embeddings
└── hnsw.bin    # serialized HNSW graph and embeddings
```

`index.json` is used by both search modes to retrieve result metadata. Exact
search compares the query with the embeddings in that file; `--hnsw` loads
`hnsw.bin` and uses the stored graph to find candidate record IDs.

If the embedding model or the text being embedded changes, run `sift ingest`
again so both files are rebuilt together.

## How It Works

Ingestion:

```text
directory
-> recursively discover supported, non-ignored source files
-> extract functions and methods with Tree-sitter
-> sort full function sources by length and embed them in batches of up to 64
-> build the exact-search records and custom HNSW graph
-> write .sift-index/index.json and .sift-index/hnsw.bin
```

Search:

```text
query
-> Jina query embedding
-> load .sift-index/index.json
-> exact cosine search (default)
   or load .sift-index/hnsw.bin and search HNSW (--hnsw)
-> print the top matches with source locations
```

Full function source is embedded for retrieval. Results remain compact by
printing only the function or method header with its source location and score.

## Roadmap

The current implementation includes a persisted custom HNSW graph and a
repeatable exact-versus-HNSW benchmark.

Planned work:

- Add evaluation datasets and identifier-aware or hybrid retrieval signals.
- Tune HNSW parameters and measure recall and latency on larger repositories.
- Replace JSON vector storage with a compact representation and support more
  efficient re-indexing.
