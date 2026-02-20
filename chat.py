"""
Interactive ChatGPT-style Q&A bot powered by OpenAI embeddings.

Usage:
    python chat.py

Requires:
    - An OpenAI API key set via the OPENAI_API_KEY environment variable
    - Pre-computed embeddings in processed/embeddings.csv
      (Run web-qa.py first to crawl and generate embeddings)
"""

import os
import sys

import numpy as np
import openai
import pandas as pd
import tiktoken
from ast import literal_eval
from openai.embeddings_utils import distances_from_embeddings

EMBEDDINGS_PATH = "processed/embeddings.csv"


def load_embeddings(path):
    """Load the pre-computed embeddings from CSV."""
    if not os.path.exists(path):
        print(f"Error: '{path}' not found.")
        print("Run web-qa.py first to crawl a website and generate embeddings.")
        sys.exit(1)

    df = pd.read_csv(path, index_col=0)
    df["embeddings"] = df["embeddings"].apply(literal_eval).apply(np.array)
    return df


def create_context(question, df, max_len=1800):
    """Find the most relevant text chunks for a question using cosine similarity."""
    q_embeddings = openai.Embedding.create(
        input=question, engine="text-embedding-ada-002"
    )["data"][0]["embedding"]

    df["distances"] = distances_from_embeddings(
        q_embeddings, df["embeddings"].values, distance_metric="cosine"
    )

    returns = []
    cur_len = 0

    for _, row in df.sort_values("distances", ascending=True).iterrows():
        cur_len += row["n_tokens"] + 4
        if cur_len > max_len:
            break
        returns.append(row["text"])

    return "\n\n###\n\n".join(returns)


def answer_question(df, question, model="text-davinci-003", max_tokens=300):
    """Answer a question based on the most similar context from the embeddings."""
    context = create_context(question, df)

    try:
        response = openai.Completion.create(
            prompt=(
                "Answer the question based on the context below, and if the "
                'question can\'t be answered based on the context, say "I don\'t know"\n\n'
                f"Context: {context}\n\n---\n\n"
                f"Question: {question}\nAnswer:"
            ),
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        return f"Error: {e}"


def main():
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY") and not openai.api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable.")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    print("Loading embeddings...")
    df = load_embeddings(EMBEDDINGS_PATH)
    print(f"Loaded {len(df)} text chunks.\n")

    print("=" * 60)
    print("  OpenAI Web Q&A Chat")
    print("  Ask questions about the crawled website content.")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 60)
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        answer = answer_question(df, question)
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()
