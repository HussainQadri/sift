# Sift

Sift is a Rust CLI prototype for semantic code search.

Given a source file and a natural-language query, it extracts functions, embeds their full source text, ranks them against the query with cosine similarity, and prints the matching function headers.

## Current usage

```bash
cargo run -- search <path> "<query>"
```

Example:

```bash
cargo run -- search src/treesitter_parse.rs "extract functions from a syntax tree"
```

After installing the binary:

```bash
sift search src/treesitter_parse.rs "extract functions from a syntax tree"
```

## Current flow

1. Parse the source file with Tree-sitter.
2. Extract function records from the syntax tree.
3. Keep each function header paired with its full source text.
4. Embed full function source with `fastembed`.
5. Embed the query string.
6. Rank function embeddings by cosine similarity.
7. Print the ranked function headers.

The full function body is used for retrieval because it gives the embedding model more context than a signature alone. The header is printed to keep results compact.

## Supported files

Current language specs:

- Rust: `.rs`
- Python: `.py`
- C++: `.cpp`

## Status

Sift currently searches one file per command and creates embeddings during that search run. It does not yet build or persist an index.

Ranking is currently vector similarity only. Code-aware ranking signals such as identifier matches, acronym handling, and token overlap are likely next steps.

