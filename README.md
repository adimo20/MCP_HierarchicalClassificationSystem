
# MCP Hierarchical Classification System 

This repository implements an MCP Server that enables RAG agents to **retrieve relevant examples and codes**, within a hierarchical classification systems in official statistics, like COICOP or NACE. When an agent retrieves relevant examples using semantic search or a keyword based search from a dataset of labelled historical examples it receives a **structured markdown summary** of the contents and meaning of a certain code within the classification system. 

The repository contains a workflow to **finetune an embedding model**, using multiple negativ ranking loss. This might be nessecary, due to the high domanin specificy of the labelled data used, e.g. heavily abreviated product names or ambigous company names. 

The MCP server also provides the agent with tools to **hierarchally search the classification system** for relevant codes, when semantic and keyword search did not lead to a relevant code.

Key features of the MCP-Server:
* Retrieval of relevant examples using semantic and keyword based search
* Hierarchical exploration of the classification system


**IMPORTANT NOTICE:** The project you find here is still in development and will be constantly append. More documentation and notebooks with examples will follow in the near future

## Table of content

- [Project Structure](#project-structure)
- [Available MCP Tools](#available-mcp-tools)
- [Components of the MCP-Server](#components-of-the-mcp-server)
  - [Loading the Classification System](#loading-the-classification-system)
    - [1. Deserializing with `from_dict()`](#1-deserializing-with-from_dict)
    - [2. Initializing the `ClassificationSystem`](#2-initializing-the-classificationsystem)
  - [ClassificationSystem](#classificationsystem)
    - [Key Features](#key-features)
    - [Internal Attributes](#internal-attributes)
    - [Core Methods](#core-methods)
    - [Hierarchical Logic Example](#hierarchical-logic-example)
  - [Markdown Augmentation Framework](#markdown-augmentation-framework)
  - [MarkdownExample](#markdownexample)
    - [Default German Taxonomy Mapping](#default-german-taxonomy-mapping)
    - [Key Methods](#key-methods)
  - [MarkdownReport](#markdownreport)
    - [Initialization & Attributes](#initialization--attributes)
    - [Key Method: `generate_markdown_summary`](#key-method-generate_markdown_summary)
  - [End-to-End Formatting Example](#end-to-end-formatting-example)
- [Retriever](#retriever)
  - [Key Features](#key-features-1)
  - [Class Initialization Attributes](#class-initial-attributes)
  - [Core Methods](#core-methods-1)
  - [Usage](#usage)
- [VectorStore & Custom Embedding Management](#vectorstore--custom-embedding-management)
  - [CustomEmbeddingFunction](#customembeddingfunction)
  - [Key Features of VectorStore](#key-features-of-vectorstore)
  - [Core Methods](#core-methods-2)
  - [Ingestion CLI Pipeline Execution](#ingestion-cli-pipeline-execution)
  - [Production Command Line Example](#production-command-line-example)
- [Model Fine-Tuning Pipeline (`MNRL.py`)](#model-fine-tuning-pipeline-mnrlpy)
  - [Key Features](#key-features-2)
  - [Process Workflow Pipeline](#process-workflow-pipeline)
  - [Command Line Arguments](#command-line-arguments)
  - [Hyperparameter Configurations](#hyperparameter-configurations)
  - [Execution Example](#execution-example)
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

  1. **`get_root_category_codes_and_descriptions`**: returns all the top level division codes for a given classification system
  2. **`get_children`**: collects and returns all the children codes for a given parent code.
  3. **`get_parent`**: collects and returns the parent codes for a given children code.
  4. **`get_code_specification`**: creates a comprehensive markdown formatted summary of the contents and meaning of a given code.

* **Relevant examples/codes retrieval**:

  5. **`semantic_search`**: performs a semantic search over a set of embedded labelled examples using ChromaDB
  6. **`full_text_search`**:perfroms a keyword based search of a set of labelled examples in side the ChromaDB

* Note: When retrieving relevants Codes using the two search methods the returned results will always be returned in a comprehensive markdown format.
---

# Compontents of the MCP-Server

## Loading the Classification System

If your classification data is saved locally as a JSON object, you can easily load and convert it using the built-in **`Code`** and **`ClassificationSystem`** classes.

### 1. Deserializing with `from_dict()`

The **`Code`** object features a `from_dict()` method, which automatically maps a standard Python dictionary (parsed from your JSON) into a structured `Code` instance.

### 2. Initializing the `ClassificationSystem`

To initialize the overarching **`ClassificationSystem`**, use a list comprehension to convert your list of dictionaries into a list of `Code` objects, then pass that list directly into the system initializer.

> **Note:** Ensure your JSON keys align exactly with the required fields (`code`, `level`, `description`, and `detailed_description`) for a seamless import. In case of missing keys, the Code obeject will leave this as empty strings.

---

## ClassificationSystem

The `ClassificationSystem` is a centralized `@dataclass` designed to manage, index, and query hierarchical classification systems (such as **COICOP** or **NACE**). It ingests a list of `Code` objects, normalizes their formats, builds an internal parent-child tree structure, and exposes optimized search and retrieval methods.

### Key Features

* **Code Normalization:** Automatically strips spaces, punctuation, and special characters (e.g., transforming `01.1.1` or `01 1 1` into `0111`). This prevents lookup failures caused by inconsistent source formatting.
* **Fast Lookups:** Indexes data into a hash map (`_lookup`) for $O(1)$ time-complexity retrievals.
* **Hierarchical Tree Mapping:** Automatically maps parent-child relationships (`_tree`) using a shared-root logic where a child's code extends its parent's code by exactly one character (e.g., Parent `01` $\rightarrow$ Child `011`).

---

### Internal Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `codes` | `list[Code]` | The raw list of `Code` objects injected into the system. |
| `_lookup` | `dict[str, Code]` | *Internal.* A preprocessed dictionary mapping normalized code strings directly to their `Code` objects. |
| `_tree` | `dict[str, list[Code]]` | *Internal.* A dictionary mapping parent codes to a list of their immediate child `Code` objects. |

---

### Core Methods

#### Retrieval & Search

* **`get_code`**
Retrieves the full details of a specific code. It normalizes the input before searching to guarantee a match regardless of formatting.
* **`get_children`**
Returns a list of all immediate child categories belonging to the specified parent code.
* **`get_code_trace`**
Traces the lineage of a given code from the top level down to itself. Returns a sequential list of `(code, description)` tuples.
> **Example:** Passing `"0111"` might return: `[("01", "Food and non-alcoholic beverages"), ("011", "Food"), ("0111", "Cereals...")]`



#### Mutation

* **`add_code(code: Code) -> None`**
Dynamically appends a new `Code` object to the active system. It automatically normalizes the code, updates the lookup index, and binds it to its parent tree.

---

### Hierarchical Logic Example

The class assumes standard tree-structured behavior where digits signify depth:

```
01 FOOD AND NON-ALCOHOLIC BEVERAGES   (Level 1 / Root Parent)
└── 011 FOOD                          (Level 2 / Child of 01)
    └── 0111 Cereals...               (Level 3 / Child of 011)

```

The system recognizes a relationship if `len(parent) + 1 == len(child)` and the child shares the parent's exact starting characters.

---

## Markdown Augmentation Framework

This framework provides an interface to convert structured data from the `ClassificationSystem` into clear, hierarchical Markdown blocks. Instead of feeding dense JSON strings to AI agents or user interfaces, this system creates highly scannable, context-rich documentation—complete with breadcrumb traces and real-world examples.

---

## MarkdownExample

The `MarkdownExample` class is the base formatting engine. It handles individual `Code` objects and styles their attributes (descriptions, hierarchy levels, and examples) into discrete Markdown sections.

> **Customization Tip:** If you adapt this repository for a different classification system (e.g., swapping from standard German taxonomy levels), this is the class you modify to change the output structure.

### Default German Taxonomy Mapping

The class initializes with an internal taxonomy tracking array to label hierarchical depths sequentially:

1. `Abteilung` (Division)
2. `Gruppe` (Group)
3. `Klasse` (Class)
4. `Unterklasse` (Subclass)
5. `Kategorie` (Category)
6. `Unterkategorie` (Subcategory)

### Key Methods

* **`header_plus_content(...) -> str`**
Generates a standardized Markdown heading and body text snippet. Allows customization of the heading weight (e.g., `##` vs `###`) and text bolding.
* **`generate_examples_part(examples: list[str]) -> str`**
Converts a Python list of strings into a formatted Markdown bulleted list under a `## Beispiele` header.
* **`format_traces_to_markdown(trace: list[tuple]) -> str`**
Accepts a code's lineage path and maps it against the internal taxonomy array.
* *Example Output:*  `Abteilung 01`: FOOD AND NON-ALCOHOLIC BEVERAGES <br>


* **`code_to_markdown(...) -> str`**
The primary orchestrator method. It aggregates the category name, code ID, detailed descriptions, structural traces, and optional examples into a single, comprehensive string.

---

## MarkdownReport

The `MarkdownReport` class acts as the bulk interface. It is responsible for opening the source JSON data, initializing the underlying lookup system, and compiling reports for multiple codes at once.

### Initialization & Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `path` | `str` | Filepath to the classification system JSON data. |
| `classification_name` | `str` | The acronym of the system being targeted (e.g., `"SEA"`, `"COICOP"`, `"NACE"`). |
| `classification` | `ClassificationSystem` | *Generated post-init.* The operational lookup system instance. |

### Key Method: `generate_markdown_summary`

This method builds a complete diagnostic report for a list of targeted codes. It matches each code to its family trace, injects relevant examples if provided, compiles the Markdown using `MarkdownExample`, and separates each entry with a clean horizontal divider (`---`).

#### Inputs:

* **`list_of_codes`**: A list of codes you want to extract and summarize (e.g., `['01111', '01112']`).
* **`examples_dict`**: An optional dictionary mapping raw code strings to lists of real-world examples (e.g., `{"01111": ["Käse", "Milch"]}`).

---

## End-to-End Formatting Example

When an AI agent or application requests information on a specific code snippet through this framework, the finalized output string renders cleanly like this:

```markdown
## Name der Kategorie

FOOD AND NON-ALCOHOLIC BEVERAGES

## SEA-Code

**01**

## Detaillierte Beschreibung

Division 01 covers food (01.1) purchased by households mainly for consumption or preparation at home and non-alcoholic beverages (01.2) purchased by households, regardless of where they are consumed. Division 01 excludes food and non-alcoholic beverages that are provided by facilities such as restaurants and school cafeterias through their food and beverage serving services (division 11).Services purchased for the processing of primary goods provided by households to produce food and non-alcoholic beverages for their own consumption are also classified under this division (01.3). Food comprises all edible goods that are purchased and consumed by households for the purpose of nourishment. Food includes: cereals and cereal products; meat; fish and other seafood; milk, other dairy products and eggs; oils and fats; fruit and nuts; vegetables, tubers, plantains, cooking bananas and pulses; sugar, confectionery and desserts; salt, condiments and sauces; and spices, culinary herbs and seeds. Division 01 also includes baby food and ready-made food that can be eaten as is or after heating. Division 01 does not include alcoholic beverages (02.1).

## Pfad der SEA-Klassifikation

`Abteilung 01`: **FOOD AND NON-ALCOHOLIC BEVERAGES**

## Beispiele

* Käse
* Milch
```
---

## Retriever

The `Retriever` class bridges the gap between historical labeled dataset storage and the structured reporting framework. It queries a **ChromaDB vector database** using either vector-based similarity or strict keyword filtering, maps the results to their respective historical taxonomy entries, cleans up format mismatches, and passes them to the `MarkdownReport` engine to synthesize context blocks for RAG agents.

### Key Features

* **Hybrid Search Modes:** Toggles between semantic vector search (`sim_search`) for domain ambiguity and text-based lookup (`text_search`) for exact string matches.
* **Smart Normalization:** Automatically strips trailing zeros via regex patterns (e.g., converting a raw DB label like `011100` down to `0111`) to preserve precise mapping consistency inside the tree logic.
* **Context Augmentation Engine:** Groups historical examples by their category classification targets, generating a centralized, evaluation-ready Markdown block natively accepted by LLMs.

---

### Class Initialization Attributes

When instantiating the `Retriever`, the system wraps a `VectorStore` instance and a `MarkdownReport` configuration simultaneously.

| Attribute | Type | Description |
| --- | --- | --- |
| `collection_name` | `str` | Name of the specific vector collection inside your ChromaDB. |
| `model_name` | `str` | **Crucial:** The explicit embedding model name used to index the text. Must exactly match the initialization model to prevent nonsensical distance calculations. |
| `path_classification_system` | `str` | Local file path to the source JSON classification system data structure. |
| `classification_name` | `str` | Target taxonomy abbreviation context (e.g., `"COICOP"`, `"NACE"`, `"SEA"`). |
| `label_key_in_collection` | `str` | The specific metadata dictionary key string where your category labels are bound inside ChromaDB documents. |
| `chromadb_path` | `str` | Directory path hosting your persistent ChromaDB client instance. |

---

### Core Methods

#### `search_collection`

Executes raw queries against ChromaDB.

* Under `sim_search`, queries the vector space with uppercase string targets.
* Under `text_search`, fallbacks to standard string evaluation using Chroma's `$contains` filter logic.

> **Performance Note:** Using `text_search` forces ChromaDB to execute a non-indexed, brute-force metadata lookup scan. For high-throughput production environments, moving text queries to a relational DB (e.g., SQLite) or a data frame architecture (Pandas/Polars) is strongly recommended.

#### `get_unique_codes`

Aggregates matching entries from the vector layer. Strips empty lookups, isolates a deduplicated sequence of target codes, and returns an ordered mapping layout indexing historical string examples to their respective classification numbers.

#### `create_augmented_context`

The primary operational entry point for RAG orchestrators. Pulls matching metadata categories, formats clean code sequences, builds a hierarchical structural layout, and produces the complete final localized Markdown prompt document.

---

### Usage

The module loads configuration states natively from project environment (.env) configurations using the layout example below:

```python
from dotenv import load_dotenv
import os
from src.retrieval.Retriever import Retriever

load_dotenv()

# Instantiate the centralized retriever agent
retriever = Retriever(
    collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
    model_name=os.getenv("CHROMA_MODEL_NAME"),
    chromadb_path=os.getenv("CHROMA_CLIENT_PATH"),
    path_classification_system=os.getenv("CHROMA_PATH_CLASSIFICATION_SYSTEM"),
    classification_name=os.getenv("CHROMA_CLASSIFICATION_NAME"),
    label_key_in_collection=os.getenv("CHROMA_LABEL_KEY_IN_COLLECTION")
)

# Extract a ready-to-inject RAG context summary block applying semantic search in the vector database
rag_prompt_context = retriever.create_augmented_context(
    q="Adidas Speziale",
    k=25,
    use_examples=True,
    search_type="sim_search"
)

print(rag_prompt_context)

```

---

## VectorStore & Custom Embedding Management

The `vector_store.py` module manages database writes, text tokenization setups, and persistence logic for the underlying **ChromaDB engine**. It handles dataset indexing pipelines by registering custom fine-tuned embedding representations and safely chunking high-volume historical inputs into atomic batches.

### CustomEmbeddingFunction

This helper component inherits from ChromaDB's core `EmbeddingFunction` abstract base class and registers itself natively within the backend execution stack via the `@register_embedding_function` decorator pattern.

* **Dual-Purpose Execution:** It coordinates two distinct lifecycle stages using a single underlying engine:
1. Compiling dense matrix vectors across baseline inputs during historical document ingestion.
2. Formatting incoming runtime text queries into identical dimensional vector shapes to ensure coherent similarity distance checks.


* **Hugging Face & Disk Compatibility:** Accepts standard local model paths (such as custom checkpoints output by your fine-tuning workflow script) or public model string identifiers hosted on the Hugging Face hub.

---

### Key Features of VectorStore

* **Thread-Safe Persistent Operations:** Uses Chroma's native storage strategy via `chromadb.PersistentClient` to preserve changes locally.
* **Automatic Collection Provisioning:** Features automated lifecycle handling to automatically instantiate new collections or latch on to historical partitions without structural overlap errors.
* **Chroma Limit Guardrails:** Breaks down massive input datasets using a built-in slicing routine to avoid exceeding Chroma's hard maximum element limits per query.

---

### Core Methods

#### `chunk_list`

Divides an abstract flat array into a nested tracking list where no segment exceeds the specific maximum parameter length. This safely spaces high-density dataset uploads across parallel operational slices.

#### `add_entries_batched`

Wraps underlying collection mutation pipelines. Configures standard operational batch increments using a strict index safety target limit threshold (`BATCH_SIZE_CHROMA = 5000`) and displays execution metrics via a visual `tqdm` console status tracker.

---

### Ingestion CLI Pipeline Execution

Executing the file directly converts a local tabular source dataset (such as `.parquet` or `.csv` files) into a structured vector layout database using simple command-line arguments.

#### Input Argument Specifications:

| Flag | Full Identifier | Data Type | Requirement / Purpose |
| --- | --- | --- | --- |
| `-f` | `--filename` | `str` | Complete path to the source `.parquet` or `.csv` training dataset file. |
| `-m` | `--model_name` | `str` | Explicit local directory string path or web checkpoint alias tracking the targeted embedding model. |
| `-c` | `--collection_name` | `str` | Name of the specific collection partition to initialize inside ChromaDB. |
| `-tc` | `--text_column` | `str` | Target data frame column containing raw string definitions or text instances meant to be vectorized. |
| `-lc` | `--label_column` | `str` | Target data frame column representing the base classification code keys (e.g., `coicop`). |

---

### Production Command Line Example

To feed custom historical mapping rows directly from an export asset down into your operational server collection layer, call the runtime handler from the root directory using the layout structure below:

```bash
python -m src.mcp_server.retrieval.vector_store \
  --filename "./data/historical_records.parquet" \
  --model_name "./models/fine_tuned_mnrl_checkpoint" \
  --collection_name "coicop_historical_v1" \
  --text_column "product_description" \
  --label_column "coicop_code"

```

---

## Model Fine-Tuning Pipeline (`MNRL.py`)

This module provides a script to fine-tune a `SentenceTransformer` embedding model optimized for domain-specific text retrieval (e.g., heavily abbreviated product names or ambiguous enterprise titles). It utilizes **Multiple Negatives Ranking Loss (MNRL)** to pull matching text-label vector spaces closer together while treating other in-batch pairs as implicit negatives (in-batch negative sampling).

Additionally, the pipeline provides experiment tracking out of the box via **MLflow**, rendering real-time loss decay slopes and information retrieval (IR) validation metrics.

---

### Key Features

* **Implicit In-Batch Negatives:** Leveraging MNRL means the pipeline doesn't require explicit negative examples. For a given batch of $(A_i, P_i)$ pairs, all positive items $P_j$ where $i \neq j$ serve as negative instances for anchor $A_i$.
* **Balanced Dataset Generation:** Combines `IterableDataset` streams with custom generator functions (`balanced_generator`) to keep dataset memory footprints minimal while preventing class imbalances from dominating batch spaces.
* **Patched MLflow Logging:** Overrides default training telemetry callbacks to seamlessly clean metric keys containing special characters (e.g., rewriting `recall@10` to `recall_at_10`), ensuring metrics rendering without backend serialization failures.

> **Dependency Constraints:** This pipeline requires `transformers==4.57.6`. Internal library updates in `transformers>=5.X.X` introduce breaking tensor formatting adjustments that interfere with correct embedding loss convergence.

---

### Process Workflow Pipeline

The execution stack is split into five isolated steps:

```text
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Parse CLI Args │ ──> │ 90/10 Train/Test Split│ ──> │ Balance & Build Loop │
└─────────────────┘     └──────────────────────┘     └──────────────────────┘
                                                                 │
                                                                 ▼
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│ MLflow Logging  │ <── │ Execute MNRL Trainer │ <── │ Setup IR Evaluator   │
└─────────────────┘     └──────────────────────┘     └──────────────────────┘

```

---

### Command Line Arguments

| Parameter | Shorthand | Type | Function |
| --- | --- | --- | --- |
| `--path_training_data_raw` | `-t` | `str` | Path to the source raw tabular `.parquet` dataset. |
| `--path_training_data_storage` | `-s` | `str` | Directory where output train/test split files are saved. |
| `--output_dir` | `-o` | `str` | Target directory where fine-tuned weights and model checkpoints are saved. |
| `--model_path` | `-m` | `str` | Local disk directory path or Hugging Face model identifier (e.g., `sentence-transformers/all-MiniLM-L6-v2`). |
| `--batch_size` | `-b` | `int` | Number of concurrent training inputs processed per device batch step. |
| `--text_column` | `-tc` | `str` | Source column name containing text/product descriptions. |
| `--label_column` | `-lc` | `str` | Source column name containing targeted classification numbers. |

---

### Hyperparameter Configurations

The script enforces a standardized baseline setup inside `SentenceTransformerTrainingArguments`:

* **Learning Rate:** `2e-5` (Optimized for fine-tuning stability without destructive rewriting of pre-trained parameters).
* **Max Steps:** `7500` training iterations.
* **Evaluation Interval:** Every `2500` steps using a dedicated information retrieval validation subset.
* **Logging Interval:** Progress updates pumped to MLflow servers every `100` steps.

---

### Execution Example

To launch a fine-tuning run locally with automated tracking, verify your local MLflow tracking server instance is online, and run the following command:

```bash
python -m src.model_training.MNRL \
  --path_training_data_raw "./data/raw_historical_records.parquet" \
  --path_training_data_storage "./data/processed_splits/" \
  --output_dir "./models/fine_tuned_coicop_model" \
  --model_path "BAAI/bge-small-en-v1.5" \
  --batch_size 64 \
  --text_column "product_name" \
  --label_column "coicop_code"

```

## Setup and Usage

### Data requirements

The MCP-Server relies on two central data sources:
  * **Documentation of a classification system** - loaded into the server in the specific json-format see section setup classification system.
  * **Vector-Database, incl. finetuned embedding model for retrieval** - If you have access to a reasonably amount of high quality annotated run the model training and embedd your historical cases using the given cli-interfaces.

Inteded workflow for set up

```text
┌─────────────────┐     ┌──────────────────────────┐     ┌─────────────────────────┐     ┌──────────────────────┐
│   Train Model   │ ──> │ Embed Historic Examples  │ ──> │ Specify Details in .env │ ──> │     Start Server     │
└─────────────────┘     └──────────────────────────┘     └─────────────────────────┘     └──────────────────────┘
```
 
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

# Author 

Adrian Montag (adrian.montag@destatis.de)