
# MCP Hierarchical Classification System 

This repository implements an MCP Server that enables RAG agents to **retrieve relevant examples and codes**, within a hierarchical classification systems, like COICOP or NACE. When an agent retrieves relevant examples using semantic search or a keyword based search from a dataset of labelled historical examples it receives a **structured markdown summary** of the contents and meaning of a certain code within the classification system. 

The repository contains a workflow to **finetune an embedding model**, using multiple negativ ranking loss. This might be nessecary, due to the high domanin specificy of the labelled data used, e.g. heavily abreviated product names or ambigous company names. 

The MCP server also provides the agent with tools to **hierarchally search the classification system** for relevant codes, when semantic and keyword search did not lead to a relevant code.

Key features of the MCP-Server:
* Retrieval of relevant examples using semantic and keyword based search
* Hierarchical exploration of the classification system


**IMPORTANT NOTICE:** The project you find here is still in development and will be constantly append. More documentation and notebooks with examples will follow in the near future


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

The `server.py` exposes the following tools to agents:

* **Hierachical Tree exploration**:

  1. **`get_root_category_codes_and_descriptions()`**: returns all the top level division codes for a given classification system
  2. **`get_children(parent_code)`**: collects and returns all the children codes for a given parent code.
  3. **`get_parent(parent_code)`**: collects and returns the parent codes for a given children code.
  4. **`get_code_specification(list_of_codes)`**: creates a comprehensive markdown formatted summary of the contents and meaning of a given code.

* **Relevant examples/codes retrieval**:

  5. **`semantic_search(q, k)`**: performs a semantic search over a set of embedded labelled examples using ChromaDB
  6. **`full_text_search(q, k)`**:perfroms a keyword based search of a set of labelled examples in side the ChromaDB

* Note: When retrieving relevants Codes using the two search methods the returned results will always be returned in a comprehensive markdown format.
---

## Setup and Usage

### Data requirements

The MCP-Server relies on two central data sources:
  * **Documentation of a classification system** - loaded into the server in a specific json-format.
  * **Vector-Database, incl. finetuned embedding model for retrieval** - If you have access to a reasonably amount of high quality annotated 

### Technical Prerequisites

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



