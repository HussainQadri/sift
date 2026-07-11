use crate::cli;
use crate::cli_output;
use crate::embeddings_generator;
use crate::hnsw;
use crate::hnsw::HnswIndex;
use crate::index;
use crate::index::IndexedFunction;
use crate::index::PersistedHnswIndex;
use crate::similarity::cosine_similarity;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub fn query_search(args: cli::Cli) -> anyhow::Result<()> {
    let keywords = match args.keywords {
        Some(query) => query,
        None => anyhow::bail!("Please enter a query."),
    };
    let top_k_results = args.top;
    let loaded_indexed_functions = index::load_index()?;

    if loaded_indexed_functions.is_empty() {
        anyhow::bail!("The index is empty, run `sift ingest <path>` first.");
    }

    let query = embeddings_generator::create_query_embedding(&keywords)?;
    if args.hnsw {
        search_using_hnsw(&query, &loaded_indexed_functions, top_k_results)?;
    } else {
        search_using_brute_force(&query, &loaded_indexed_functions, top_k_results)?;
    }
    Ok(())
}

pub fn search_using_hnsw(
    query: &[f32],
    loaded_indexed_functions: &[IndexedFunction],
    top_k_results: usize,
) -> anyhow::Result<()> {
    // Read hnsw.bin, deserialize, convert each persisted node to runtime node
    // Assign Node.id from node's vector position
    // Copy graph config into runtime HnswIndex

    let bytes = fs::read(index::HNSW_INDEX_PATH)?;
    let deserialised_hnsw_graph: PersistedHnswIndex = postcard::from_bytes(&bytes)?;

    let mut index = HnswIndex::new(32, 256);

    // Reassign entry_point, ef, m and max_layer to runtime index
    index.entry_point = deserialised_hnsw_graph.entry_point;
    index.ef = deserialised_hnsw_graph.ef;
    index.m = deserialised_hnsw_graph.m;
    index.max_layer = deserialised_hnsw_graph.max_layer;

    // Now go through nodes in deserialised hnsw, construct runtime Nodes
    for (persisted_node_pos, persisted_node) in
        deserialised_hnsw_graph.nodes.into_iter().enumerate()
    {
        let runtime_node: hnsw::Node = hnsw::Node {
            embedding: persisted_node.embedding,
            id: persisted_node_pos,
            neighbours: persisted_node.neighbours,
            record_id: persisted_node.record_id,
        };

        index.nodes.push(runtime_node);
    }

    println!("HNSW OUTPUT");
    let records_by_id: HashMap<usize, &index::IndexedFunction> = loaded_indexed_functions
        .iter()
        .map(|record| (record.record_id, record))
        .collect();
    // These are record_ids from the JSON not the internal HNSW index
    let result_ids = index.search(query, top_k_results);
    for record_id in result_ids {
        let indexed_function = match records_by_id.get(&record_id) {
            Some(value) => value,
            None => {
                eprintln!("HNSW returned unknown record id: {record_id}");
                continue;
            }
        };
        let score = cosine_similarity(query, &indexed_function.embedding);

        println!("{:.3} {}: ", score, indexed_function.path);

        let extension = Path::new(&indexed_function.path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        print!("\x1b[32m{}:\x1b[0m ", indexed_function.line_number);
        cli_output::print_highlighted(&indexed_function.header, extension);
        println!("\n");
    }
    Ok(())
}
pub fn search_using_brute_force(
    query: &[f32],
    loaded_indexed_functions: &[IndexedFunction],
    top_k_results: usize,
) -> anyhow::Result<()> {
    println!("BRUTE FORCE OUTPUT");
    let mut result: Vec<(&index::IndexedFunction, f32)> = loaded_indexed_functions
        .iter()
        .map(|indexed_function| {
            let score = cosine_similarity(query, &indexed_function.embedding);

            (indexed_function, score)
        })
        .collect();
    result.sort_by(|a, b| b.1.total_cmp(&a.1));

    for (indexed_function, score) in result.iter().take(top_k_results) {
        println!("{:.3} {}: ", score, indexed_function.path);
        let extension = Path::new(&indexed_function.path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        print!("\x1b[32m{}:\x1b[0m ", indexed_function.line_number);
        cli_output::print_highlighted(&indexed_function.header, extension);
        println!("\n");
    }
    Ok(())
}
