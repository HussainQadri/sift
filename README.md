# Sift

![CI](https://github.com/HussainQadri/sift/actions/workflows/ci.yml/badge.svg)

Sift is local semantic code search for codebases. Ask in natural language and
jump straight to the functions that matter.

<p align="center">
  <img src="assets/sift-demo.gif" alt="Sift CLI demo">
</p>

Sift uses Tree-sitter to extract functions and methods, embeds their source with
the Potion Code 16M v2 static embedding model (`minishlab/potion-code-16M-v2`),
and stores the resulting 256-dimensional vectors locally. The custom persisted
HNSW index provides the default search mode; exhaustive cosine search is
available when exact results are required.

Embedding inference and indexing run locally. On first use, model2vec downloads
the model into the HuggingFace cache directory. File discovery and Tree-sitter
parsing run in parallel across CPU cores, and embedding uses a single shared
model session; no GPU is required. Function sources and queries are truncated
to 256 tokens.

## Install

Sift requires Rust 1.85 or newer (the crate uses edition 2024). From this
repository, run:

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

Search using the HNSW index:

```bash
sift "load saved index records"
```

Return a different number of results (the default is 3):

```bash
sift --top 5 "load saved index records"
```

Use exhaustive cosine similarity instead:

```bash
sift --exact --top 5 "load saved index records"
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

## Evaluating Retrieval Quality

After ingesting a codebase, create a JSON file describing judged queries. Each
entry pairs a natural-language query with the functions that should be
retrieved, graded from 1 (marginally relevant) to 3 (highly relevant):

```json
[
  {
    "query": "compute cosine similarity between two vectors",
    "relevant": [
      {
        "path": "src/similarity.rs",
        "header": "pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32",
        "relevance": 3
      }
    ]
  }
]
```

Then measure average nDCG@k over the judgements:

```bash
sift evaluate --judgements judgements.json --top 10
```

A judgement matches a search result when the header is identical and the
indexed path ends with the judgement's path. `--top` defaults to 10. Evaluation
always uses exhaustive search so the score measures retrieval quality rather
than HNSW approximation.

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

`index.json` is used by both search modes to retrieve result metadata. Default
search loads `hnsw.bin` to find candidate record IDs; `--exact` compares the
query exhaustively with the embeddings stored in `index.json`.

If the embedding model or the text being embedded changes, run `sift ingest`
again so both files are rebuilt together.

## How It Works

Ingestion:

```text
directory
-> recursively discover supported, non-ignored source files in parallel
-> extract functions and methods with Tree-sitter
-> sort function sources by length with deterministic path/line tie-breaks
-> embed sources with a single shared model session in batches of 1024
-> build the exact-search records and custom HNSW graph
-> write .sift-index/index.json and .sift-index/hnsw.bin
```

Search:

```text
query
-> Potion query embedding
-> load .sift-index/index.json
-> load .sift-index/hnsw.bin and search HNSW (default)
   or run exhaustive cosine search over every record (--exact)
-> print the top matches with source locations
```

Function source is embedded for retrieval, truncated to 256 tokens. Results
remain compact by printing only the function or method header with its source
location and score.

## Roadmap

The current implementation includes parallel file discovery and parsing, a
persisted custom HNSW graph, an exact-versus-HNSW benchmark, and graded
evaluation with nDCG@k.

Planned work:

- Add identifier-aware or hybrid retrieval signals.
- Tune HNSW parameters and measure recall and latency on larger repositories.
- Replace JSON vector storage with a compact representation and support more
  efficient re-indexing.
