from collections import Counter
from pathlib import Path
import re

from wordcloud import WordCloud


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def main():
    base_dir = Path(__file__).resolve().parent
    corpus_path = base_dir / "corpus.txt"
    output_path = base_dir / "corpus_wordcloud.png"

    text = corpus_path.read_text(encoding="utf-8")
    tokens = tokenize(text)
    token_counts = Counter(tokens)

    total_tokens = len(tokens)
    vocab_size = len(token_counts)

    print(f"Corpus file: {corpus_path}")
    print(f"Total tokens: {total_tokens}")
    print(f"Vocabulary size: {vocab_size}")

    wordcloud = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        colormap="viridis",
    ).generate_from_frequencies(token_counts)
    wordcloud.to_file(str(output_path))
    print(f"Word cloud saved to: {output_path}")


if __name__ == "__main__":
    main()