use syntect::{
    easy::HighlightLines,
    highlighting::{Style, ThemeSet},
    parsing::SyntaxSet,
    util::{LinesWithEndings, as_24_bit_terminal_escaped},
};

pub fn print_highlighted(code: &str, extension: &str) {
    let syntax_set = SyntaxSet::load_defaults_newlines();
    let theme_set = ThemeSet::load_defaults();

    let syntax = syntax_set
        .find_syntax_by_extension(extension)
        .unwrap_or_else(|| syntax_set.find_syntax_plain_text());

    let theme = &theme_set.themes["base16-ocean.dark"];

    let mut highlighter = HighlightLines::new(syntax, theme);

    for line in LinesWithEndings::from(code) {
        let ranges: Vec<(Style, &str)> = highlighter.highlight_line(line, &syntax_set).unwrap();

        print!("{}", as_24_bit_terminal_escaped(&ranges[..], false));
    }
}
