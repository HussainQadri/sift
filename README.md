# Sift

![CI](https://github.com/HussainQadri/sift/actions/workflows/ci.yml/badge.svg)

Sift is local semantic code search for codebases. Ask in natural language and
jump straight to the functions that matter.

<p align="center">
  <img src="assets/sift-demo.gif" alt="Sift CLI demo">
</p>

Sift uses Tree-sitter to extract functions and methods, embeds their full source
with the quantized Snowflake Arctic Embed XS model, and stores the resulting
384-dimensional vectors locally. The custom persisted HNSW index provides the
default search mode; exhaustive cosine search is available when exact results
are required.

Embedding inference and indexing run locally. On first use, FastEmbed downloads
the model into the operating system's cache directory. Ingestion uses
multiple CPU model sessions concurrently; no GPU is required.

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

## Ingestion Performance

Release-mode benchmarks on an Intel i7-9700K with eight physical cores and a
cached model produced the following results:

| Repository | Functions | Serial ingest | Parallel ingest | Speedup |
|---|---:|---:|---:|---:|
| ripgrep | 2,744 | 16.1 s | 10.52 s | 1.53x |
| rust-analyzer | 23,734 | 170.78 s | 123.46 s | 1.38x |

The rust-analyzer figures are medians across five runs. Parallel ingestion
reduced elapsed time by 27.7%, saving 47.32 seconds per ingest. Peak memory rose
from approximately 430 MiB to 741 MiB because multiple model sessions are held
in memory concurrently. All 23,734 serial and parallel records had identical
metadata and embedding vectors.

## How It Works

Ingestion:

```text
directory
-> recursively discover supported, non-ignored source files
-> extract functions and methods with Tree-sitter
-> sort full function sources by length with deterministic path/line tie-breaks
-> embed full function sources concurrently across independent CPU model sessions
-> build the exact-search records and custom HNSW graph
-> write .sift-index/index.json and .sift-index/hnsw.bin
```

Search:

```text
query
-> Snowflake Arctic Embed XS query embedding
-> load .sift-index/index.json
-> load .sift-index/hnsw.bin and search HNSW (default)
   or run exhaustive cosine search over every record (--exact)
-> print the top matches with source locations
```

Full function source is embedded for retrieval. Results remain compact by
printing only the function or method header with its source location and score.

## Roadmap

The current implementation includes parallel CPU embedding, a persisted custom
HNSW graph, and a repeatable exact-versus-HNSW benchmark.

Planned work:

- Add evaluation datasets and identifier-aware or hybrid retrieval signals.
- Tune HNSW parameters and measure recall and latency on larger repositories.
- Replace JSON vector storage with a compact representation and support more
  efficient re-indexing.
