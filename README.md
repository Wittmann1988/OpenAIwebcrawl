# Web Q&A with Embeddings

Learn how to crawl your website and build a Q/A bot with the OpenAI API. You can find the full tutorial in the [OpenAI documentation](https://platform.openai.com/docs/tutorials/web-qa-embeddings).

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY='sk-...'
```

### 3. Crawl a website and generate embeddings

```bash
python web-qa.py
```

This crawls `openai.com`, extracts text, and creates embeddings stored in `processed/embeddings.csv`.

### 4. Start the interactive chat

```bash
python chat.py
```

This opens a ChatGPT-style chat interface where you can ask questions about the crawled content. Type `quit` or `exit` to end the session.
