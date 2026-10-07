use crate::index::IndexedFunction;
pub fn tokenize(text: &str) -> anyhow::Result<Vec<String>> {
    let mut cur_word = String::new();
    let mut tokenized_result = Vec::new();
    for char in text.chars() {
        if char.is_alphabetic() || char.is_digit(10) || char.is_alphanumeric() {
            cur_word.push(char);
        } else {
            if !cur_word.is_empty() {
                cur_word = cur_word.to_lowercase();
                tokenized_result.push(std::mem::take(&mut cur_word));
            }
        }
    }
    Ok(tokenized_result)
}

/// This function finds the frequency of the i'th query term
fn calculate_f_i(
    tokenized_query: &Vec<String>,
    i: usize,
    indexed_functions: &Vec<IndexedFunction>,
) -> anyhow::Result<usize> {
    anyhow::ensure!(
        i < tokenized_query.len(),
        "i'th term of query required but i is out of bounds"
    );

    let mut freq = 0;
    let term_needed = tokenized_query.get(i).unwrap();
    // Go through indexed functions, check if term needed is inside each indexed function and
    // calculate frequency
    for indexed_function in indexed_functions {
        freq += tokenize(&indexed_function.header)?
            .iter()
            .filter(|token| *token == term_needed)
            .count()
            + tokenize(&indexed_function.source)?
                .iter()
                .filter(|token| *token == term_needed)
                .count()
    }

    Ok(freq)
}

pub fn idf(doc_count: usize) {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn check_basic_tokenization() {
        let basic_query = String::from("where is error logging done?");
        let res = tokenize(&basic_query).unwrap();
        assert_eq!(res, vec!["where", "is", "error", "logging", "done"]);
    }

    #[test]
    fn get_frequency_for_term_in_query_with_documents() {
        let query = String::from("error logging");
        let tokenized_query = tokenize(&query).unwrap();

        let indexed_functions = vec![
            IndexedFunction {
                path: String::from("src/a.rs"),
                header: String::from("fn log_error(msg: &str)"),
                source: String::from("fn log_error(msg: &str) { println!(\"error: {}\", msg); }"),
                line_number: 1,
                embedding: vec![],
                record_id: 0,
            },
            IndexedFunction {
                path: String::from("src/b.rs"),
                header: String::from("fn parse(input: &str)"),
                source: String::from("fn parse(input: &str) { let error = input; }"),
                line_number: 10,
                embedding: vec![],
                record_id: 1,
            },
            IndexedFunction {
                path: String::from("src/c.rs"),
                header: String::from("fn main()"),
                source: String::from("fn main() { let x = 1; }"),
                line_number: 20,
                embedding: vec![],
                record_id: 2,
            },
        ];

        assert_eq!(
            calculate_f_i(&tokenized_query, 0, &indexed_functions).unwrap(),
            4
        );
    }
}
