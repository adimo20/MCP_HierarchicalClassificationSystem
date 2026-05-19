```markdown
# MCP Hierarchical Classification System 
This repository contains a Model Context Protocol (MCP) server and an AI agent designed to automatically classify products, services, and household expenses according to the **SEA (Systematik der Einnahmen und Ausgaben der Privaten Haushalte)**. 

The project leverages a combination of semantic vector search, full-text retrieval, hierarchical traversal tools, and a DSPy-powered ReAct agent to act as an expert classification auditor with zero-hallucination constraints.

---

## Key Features

* **FastMCP Server**: Exposes a robust suite of tools via Server-Sent Events (SSE) for querying and navigating the SEA classification tree.
* **Hybrid Retrieval (ChromaDB)**: 
  * *Semantic Search:* Uses fine-tuned SentenceTransformer embeddings to classify complex or descriptive items.
  * *Full-Text Search:* Uses exact keyword matching for standardized nouns.
* **Hierarchical Navigation**: Built-in logic to drill down into child categories or abstract up to parent categories when user input is ambiguous.
* **DSPy Agent Integration**: An advanced ReAct agent that strictly adheres to Standard Operating Procedures (SOPs) to ensure factual, traceable, and accurate classifications.
* **Custom Model Training**: Includes a pipeline (`MNRL.py`) for fine-tuning SentenceTransformer models using Multiple Negatives Ranking Loss (MNRL) and tracking experiments with MLflow.

---

## Project Structure

```text
├── agent.py                            # DSPy ReAct agent that interacts with the MCP Server
├── src/
│   ├── mcp_server/
│   │   ├── server.py                   # FastMCP server definition and tool registration
│   │   ├── classification_system/      # Core logic for handling SEA hierarchy (Codes, Trees)
│   │   │   ├── classification_system.py 
│   │   │   └── MarkdownAugmentation.py 
│   │   └── retrieval/                  # ChromaDB vector store and hybrid search implementation
│   │       ├── Retriever.py
│   │       └── vector_store.py
│   └── model_training/                 # Scripts to fine-tune the embedding model
│       ├── MNRL.py                     # Training script using Multiple Negatives Ranking Loss
│       └── helpers/                    # Dataset generation and evaluation scripts

```

---

## Available MCP Tools

The `server.py` exposes the following tools to large language models or agents:

1. **`get_root_category_codes_and_descriptions`**: Returns the top-level divisions (root categories) of the SEA classification system. Used as a starting point for entirely new items.
2. **`get_children(parent_code)`**: Collects direct child categories for a given parent code to drill down into the hierarchy.
3. **`get_parent(parent_code)`**: Retrieves the immediate parent category. Used for abstraction when a search finds a highly specific code but the input lacks sufficient detail to justify it.
4. **`get_code_specification(list_of_codes)`**: Generates a definitive Markdown report detailing the official rules, inclusions, exclusions, and hierarchical trace for specific codes.
5. **`semantic_search(q, k)`**: Performs a natural language similarity search over historical data to suggest SEA codes for complex/descriptive strings (e.g., "Organic Whole Grain Wheat Bread").
6. **`full_text_search(q, k)`**: Performs exact keyword matching over historical data for short, exact nouns (e.g., "Jeans", "Milch").

---

## Setup and Usage

### Prerequisites

* Python 3.10+
* A running instance of ChromaDB or a local persistent path setup.
* `.env` file configured with your local environment variables.

### Environment Variables (`.env`)

You must define the following variables in a `.env` file in the root directory:

```env
# MCP Server / ChromaDB Config
CHROMA_COLLECTION_NAME=your_collection_name
CHROMA_MODEL_NAME=your_embedding_model_name
CHROMA_PATH_CLASSIFICATION_SYSTEM=path/to/sea_classification.json
CHROMA_CLASSIFICATION_NAME=SEA
CHROMA_LABEL_KEY_IN_COLLECTION=coicop
CHROMA_CLIENT_PATH=path/to/chromadb

# Agent Config
SERVER_URL_=http://localhost:8080/sse
MODEL_NAME=your_llm_model  # e.g., openai/gpt-4o
API_BASE=your_api_base
API_KEY=your_api_key

# Model Training / MLflow Config
ML_FLOW_URI=[http://127.0.0.1:5000](http://127.0.0.1:5000)
MODEL_FINETUNING_EXPERIMENT=Retrieval_Model_Training

```

### 1. Running the MCP Server

Start the FastMCP server, which will listen for SSE connections on port 8080:

```bash
python src/mcp_server/server.py

```

### 2. Running the DSPy Agent

Once the server is running, you can prompt the agent to classify a specific expense or product:

```bash
python agent.py "Bio Vollkornbrot mit Sonnenblumenkernen"

```

The agent will iteratively use the tools (Semantic Search -> Retrieve Specifications -> Drill down/Abstract) and output a thought process followed by the exact SEA code and justification.

---

## Model Training

To improve semantic search accuracy, you can fine-tune the `sentence-transformers` embedding model on your historic classification data.

Run the training pipeline utilizing `MNRL.py`:

```bash
python src/model_training/MNRL.py \
  --path_training_data_raw "data/raw_data.parquet" \
  --path_training_data_storage "data/processed/" \
  --output_dir "models/fine-tuned-model" \
  --model_path "sentence-transformers/all-MiniLM-L6-v2" \
  --batch_size 32 \
  --text_column "product_description" \
  --label_column "sea_code"

```

*Note: The script integrates natively with MLflow to track parameters, evaluation recall, and loss metrics.*

```

```