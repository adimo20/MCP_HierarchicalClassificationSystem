# MCP Hierarchical Classification System

This repository implements an MCP Server that enables RAG agents to **retrieve relevant examples and codes** within a hierarchical classification system in official statistics, like COICOP or NACE. When an agent retrieves relevant examples using semantic search or a keyword-based search from a dataset of labelled historical examples, it receives a **structured markdown summary** of the contents and meaning of a certain code within the classification system.

The repository contains a workflow to **finetune an embedding model** using Multiple Negatives Ranking Loss. This may be necessary due to the high domain specificity of the labelled data used, e.g. heavily abbreviated product names or ambiguous company names.

The MCP server also provides the agent with tools to **hierarchically search the classification system** for relevant codes when semantic and keyword search did not lead to a relevant result.

Key features of the MCP Server:
* Retrieval of relevant examples using semantic and keyword-based (SQLite FTS5) search
* Hierarchical exploration of the classification system
* Structured Markdown context generation for RAG pipelines
* Dependency-injected, testable architecture

**IMPORTANT NOTICE:** This project is still in development and will be constantly expanded. More documentation and notebooks with examples will follow.

## Table of Contents

- [Project Structure](#project-structure)
- [Available MCP Tools](#available-mcp-tools)
- [Components of the MCP Server](#components-of-the-mcp-server)
  - [Loading the Classification System](#loading-the-classification-system)
  - [ClassificationSystem](#classificationsystem)
  - [Markdown Augmentation Framework](#markdown-augmentation-framework)
  - [MarkdownExample](#markdownexample)
  - [MarkdownReport](#markdownreport)
  - [End-to-End Formatting Example](#end-to-end-formatting-example)
- [Retriever](#retriever)
- [StringMatcher](#stringmatcher)
- [VectorStore & Custom Embedding Management](#vectorstore--custom-embedding-management)
- [Description Generation Pipeline](#description-generation-pipeline)
- [Model Fine-Tuning Pipeline (MNRL.py)](#model-fine-tuning-pipeline-mnrlpy)
- [Agents](#agents)
- [Testing](#testing)
- [Setup and Usage](#setup-and-usage)

---

## Project Structure

```text
├── agent.py                                  # DSPy ReAct agent that interacts with the MCP Server via SSE
├── src/
│   ├── mcp_server/
│   │   ├── server.py                         # FastMCP server: dependency injection + tool registration
│   │   ├── agents/
│   │   │   └── online_retail_classification.py  # Standalone DSPy ReAct agent (no MCP, direct tool calls)
│   │   ├── classification_system/
│   │   │   ├── classification_system.py      # Code dataclass and ClassificationSystem tree logic
│   │   │   ├── MarkdownAugmentation.py       # MarkdownExample and MarkdownReport formatters
│   │   │   └── description_generation/
│   │   │       ├── DescriptionGenerationPipeline.py   # Orchestrates LLM-based description generation
│   │   │       ├── label_generation/
│   │   │       │   └── label_augmentation.py          # LabelDescriptionGenerator (bottom-up LLM calls)
│   │   │       └── dataloaders/
│   │   │           ├── CoicopDataloader.py            # Downloads/parses COICOP XLSX from UN
│   │   │           └── KlassServerDataloader.py       # Parses SEA/WZ XML from Klass server
│   │   └── retrieval/
│   │       ├── Retriever.py                  # Hybrid search + Markdown context builder
│   │       ├── StringMatcher.py              # SQLite FTS5 full-text search backend
│   │       └── vector_store.py               # ChromaDB wrapper + custom embedding function
│   ├── model_training/
│   │   ├── MNRL.py                           # Training script using Multiple Negatives Ranking Loss
│   │   └── helpers/
│   │       ├── dataset_generator.py          # Balanced (anchor, positive) pair generator
│   │       └── InformationRetrievalEval.py   # IR evaluator setup for sentence-transformers
│   └── tests/
│       └── unit/
│           └── test_classification_system.py # Pytest suite for ClassificationSystem and Markdown formatters
```

---

## Available MCP Tools

`server.py` exposes the following tools to agents:

**Hierarchical tree exploration:**

1. **`get_root_category_codes_and_descriptions`** — returns all top-level division codes and descriptions for the classification system.
2. **`get_children`** — returns all direct child codes for a given parent code.
3. **`get_parent`** — returns the immediate parent code for a given child code. Used for abstraction when input lacks the granularity to justify a leaf node.
4. **`get_code_specification`** — produces a comprehensive Markdown summary of the contents, path, and meaning of one or more given codes.

**Relevant examples/codes retrieval:**

5. **`semantic_search`** — performs a vector similarity search over embedded historical examples using ChromaDB.
6. **`full_text_search`** — performs a keyword-based FTS5 search over historical examples stored in SQLite.

Results from both search tools are always returned as structured Markdown, ready for injection into an LLM prompt.

---

## Components of the MCP Server

### Loading the Classification System

Classification data saved locally as a JSON array can be loaded using the built-in `Code` and `ClassificationSystem` classes.

**1. Deserializing with `from_dict()`**

The `Code` dataclass exposes a `from_dict()` classmethod that maps a standard Python dictionary into a structured `Code` instance. Keys not present in the source dictionary are left as empty defaults.

**2. Initializing the `ClassificationSystem`**

Pass a list comprehension of `Code` objects directly to the `ClassificationSystem` initializer:

```python
with open("sea_classification.json", "r", encoding="utf-8") as f:
    data = json.load(f)
codes = [Code.from_dict(c) for c in data]
system = ClassificationSystem(codes=codes)
```

> **Note:** Ensure your JSON keys align with the required fields (`code`, `level`, `description`, `detailled_description`). Missing keys are silently left as empty strings.

---

### ClassificationSystem

The `ClassificationSystem` is a centralized `@dataclass` that ingests a list of `Code` objects, normalizes their formats, builds an internal parent–child tree, and exposes optimized search and retrieval methods.

**Key Features**

* **Code Normalization:** Strips spaces, punctuation, and special characters (e.g., `01.1.1` or `01 1 1` → `0111`), preventing lookup failures from inconsistent source formatting.
* **Fast Lookups:** Indexes codes into a hash map (`_lookup`) for O(1) retrieval.
* **Hierarchical Tree Mapping:** Automatically maps parent–child relationships (`_tree`) using shared-root logic where a child's code extends its parent's by exactly one character (e.g., parent `01` → child `011`).

**Internal Attributes**

| Attribute | Type | Description |
| --- | --- | --- |
| `codes` | `list[Code]` | The raw list of `Code` objects injected at initialization. |
| `_lookup` | `dict[str, Code]` | Preprocessed hash map of normalized code strings to `Code` objects. |
| `_tree` | `dict[str, list[Code]]` | Maps each parent code to its list of immediate child `Code` objects. |

**Core Methods**

* **`get_code(code)`** — retrieves a `Code` object by string, normalizing input before lookup.
* **`get_children(parent)`** — returns a list of all immediate child `Code` objects for the given parent.
* **`get_code_trace(code)`** — traces the full lineage from the root down to the given code, returning a list of `(code, description)` tuples.
* **`add_code(code)`** — dynamically appends a new `Code` to the live system, updating both `_lookup` and `_tree`.

**Hierarchical Logic Example**

```
01 FOOD AND NON-ALCOHOLIC BEVERAGES   (Level 1 / Root)
└── 011 FOOD                          (Level 2 / Child of 01)
    └── 0111 Cereals...               (Level 3 / Child of 011)
```

A parent–child relationship is recognized when `len(parent) + 1 == len(child)` and the child shares the parent's exact starting characters.

---

### Markdown Augmentation Framework

This framework converts structured data from the `ClassificationSystem` into clear, hierarchical Markdown blocks. Rather than feeding dense JSON to agents, it produces scannable, context-rich documentation including breadcrumb traces and real-world examples.

---

### MarkdownExample

The `MarkdownExample` class is the base formatting engine. It handles individual `Code` objects and styles their attributes into discrete Markdown sections.

> **Customization:** To adapt this repository for a different classification system, modify this class to change how codes are presented to agents.

**Default German Taxonomy Mapping**

The class initializes with an internal taxonomy array for labelling hierarchical depths:

1. `Abteilung` (Division)
2. `Gruppe` (Group)
3. `Klasse` (Class)
4. `Unterklasse` (Subclass)
5. `Kategorie` (Category)
6. `Unterkategorie` (Subcategory)

**Key Methods**

* **`header_plus_content`** — generates a Markdown heading and body snippet; heading weight and content bolding are configurable.
* **`generate_examples_part`** — converts a list of strings into a Markdown bulleted list under a `## Beispiele` header.
* **`format_traces_to_markdown`** — maps a code's lineage path against the taxonomy array. Example output: `` `Abteilung 01`: **FOOD AND NON-ALCOHOLIC BEVERAGES** <br> ``
* **`code_to_markdown`** — the primary orchestrator. Assembles category name, code ID, detailed description, structural trace, and optional examples into a single comprehensive string.

---

### MarkdownReport

`MarkdownReport` is the bulk interface. It opens the source JSON, initializes the underlying `ClassificationSystem`, and compiles Markdown reports for sets of codes at once.

**Initialization Attributes**

| Attribute | Type | Description |
| --- | --- | --- |
| `path` | `str` | Filepath to the classification system JSON. |
| `classification_name` | `str` | Acronym of the system (e.g., `"SEA"`, `"COICOP"`, `"NACE"`). |
| `classification` | `ClassificationSystem` | Generated post-init; the operational lookup system. |

**Key Method: `generate_markdown_summary`**

Builds a complete diagnostic report for a list of target codes. It matches each code to its family trace, injects relevant examples if provided, compiles Markdown via `MarkdownExample`, and separates entries with horizontal dividers (`---`).

Inputs:
* **`list_of_codes`** — codes to summarize (e.g., `['01111', '01112']`).
* **`examples_dict`** — optional dict mapping code strings to lists of real-world examples (e.g., `{"01111": ["Käse", "Milch"]}`). Defaults to `None` (no examples shown).

---

### End-to-End Formatting Example

```markdown
## Name der Kategorie

FOOD AND NON-ALCOHOLIC BEVERAGES

## SEA-Code

**01**

## Detaillierte Beschreibung

Division 01 covers food (01.1) purchased by households mainly for consumption or
preparation at home and non-alcoholic beverages (01.2)...

## Pfad der SEA-Klassifikation

`Abteilung 01`: **FOOD AND NON-ALCOHOLIC BEVERAGES**

## Beispiele

* Käse
* Milch
```

---

## Retriever

The `Retriever` class bridges historical labelled data storage and the structured reporting framework. It queries ChromaDB or SQLite, maps results to their taxonomy entries, and builds Markdown context blocks for RAG agents.

**Architecture: Dependency Injection**

The `Retriever` no longer constructs its dependencies internally. It receives pre-instantiated `VectorStore`, `StringMatcher`, and `MarkdownReport` objects at initialization. This makes it easier to test and compose in `server.py`.

```python
retriever = Retriever(
    label_key_in_collection=os.getenv("CHROMA_LABEL_KEY_IN_COLLECTION"),
    vector_store=vs,
    string_matcher=matcher,
    classification_system=classification_system
)
```

**Key Features**

* **Hybrid Search Modes:** Toggles between `sim_search` (ChromaDB vector similarity) and `text_search` (SQLite FTS5 via `StringMatcher`).
* **Smart Normalization:** Strips trailing zeros from raw DB labels (e.g., `011100` → `0111`) to maintain accurate tree-mapping.
* **Context Augmentation:** Groups historical examples by category code and generates a Markdown block ready for LLM injection.

**Initialization Attributes**

| Attribute | Type | Description |
| --- | --- | --- |
| `label_key_in_collection` | `str` | Metadata key under which labels are stored in ChromaDB documents. |
| `vector_store` | `VectorStore` | Pre-instantiated ChromaDB wrapper. |
| `string_matcher` | `StringMatcher` | Pre-instantiated SQLite FTS5 search backend. |
| `classification_system` | `MarkdownReport` | Pre-instantiated Markdown report generator. |

**Core Methods**

* **`search_collection(q, k)`** — queries ChromaDB for the `k` most similar documents to `q`.
* **`get_unique_codes(q, k, label_key)`** — aggregates and deduplicates codes from vector search results, returning both a code list and a `{code: [examples]}` dict.
* **`create_augmented_context(q, k, use_examples, search_type)`** — the main entry point for RAG orchestrators. Dispatches to `sim_search` or `text_search`, normalizes codes, and returns a full Markdown prompt context.

**Usage**

```python
from dotenv import load_dotenv
import os
from src.mcp_server.retrieval.Retriever import Retriever
from src.mcp_server.retrieval.vector_store import VectorStore
from src.mcp_server.retrieval.StringMatcher import StringMatcher
from src.mcp_server.classification_system.MarkdownAugmentation import MarkdownReport

load_dotenv()

vs = VectorStore(
    collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
    model_name=os.getenv("CHROMA_MODEL_NAME"),
    chromadb_path=os.getenv("CHROMA_CLIENT_PATH")
)
matcher = StringMatcher(
    path_to_df=os.getenv("PATH_TO_DF"),
    path_sqlite=os.getenv("PATH_SQLITE"),
    text_column=os.getenv("TEXT_COLUMN"),
    label_column=os.getenv("CHROMA_LABEL_KEY_IN_COLLECTION"),
    table_name=os.getenv("TABLE_NAME")
)
classification_system = MarkdownReport(
    path=os.getenv("CHROMA_PATH_CLASSIFICATION_SYSTEM"),
    classification_name=os.getenv("CHROMA_CLASSIFICATION_NAME"),
)

retriever = Retriever(
    label_key_in_collection=os.getenv("CHROMA_LABEL_KEY_IN_COLLECTION"),
    vector_store=vs,
    string_matcher=matcher,
    classification_system=classification_system
)

rag_context = retriever.create_augmented_context(
    q="Adidas Speziale",
    k=25,
    use_examples=True,
    search_type="sim_search"
)
print(rag_context)
```

---

## StringMatcher

`StringMatcher` provides fast keyword-based full-text search over historical labelled examples using **SQLite FTS5**. It is used as the backend for the `full_text_search` MCP tool and as an alternative to vector search in the `Retriever`.

**Key Features**

* **FTS5 Virtual Table:** On initialization, loads the source file (`.csv` or `.parquet`) into an FTS5 virtual table for optimized full-text queries.
* **Two-Stage Search:** First attempts an exact match (`WHERE klartext='...'`), then falls back to a substring/token match (`WHERE klartext MATCH '...'`).
* **Label Normalization:** Applies the same trailing-zero stripping and special-character removal as the `ClassificationSystem` before inserting and querying labels.

> **Performance Note:** FTS5 is well-suited for moderate dataset sizes. For very high-throughput production environments, consider a dedicated search backend.

**Initialization Attributes**

| Attribute | Type | Description |
| --- | --- | --- |
| `path_to_df` | `str` | Path to the source `.csv` or `.parquet` file. |
| `path_sqlite` | `str` | Path where the SQLite database will be created/loaded. |
| `text_column` | `str` | Column name containing the text examples to search. |
| `label_column` | `str` | Column name containing the classification codes. |
| `table_name` | `str` | Name of the FTS5 virtual table inside SQLite. |

**Core Methods**

* **`match_data(q, k_per_class)`** — runs the two-stage search and returns `(unique_labels, {label: [examples]})`, or `(None, None)` if no results are found.
* **`organise_data(query_results, num_examples_cap)`** — groups raw query results into a `{label: [examples]}` dict, capping each class at `num_examples_cap` entries.

---

## VectorStore & Custom Embedding Management

`vector_store.py` manages ChromaDB writes, custom embedding registration, and persistence logic.

**CustomEmbeddingFunction**

Inherits from ChromaDB's `EmbeddingFunction` base class and registers itself via `@register_embedding_function`. Wraps a `SentenceTransformer` model for both document ingestion and query encoding.

* Accepts local model paths (e.g., fine-tuned checkpoints) or Hugging Face model identifiers.

**Key Features of VectorStore**

* **Persistent Client:** Uses `chromadb.PersistentClient` to store data locally on disk.
* **Automatic Collection Provisioning:** Creates or retrieves an existing collection without overlap errors via `get_or_create_collection`.
* **Batch Guardrails:** Chunks large datasets into slices of 5,000 to stay within ChromaDB's per-request limits.

**Core Methods**

* **`chunk_list(list_to_chunk, chunk_size)`** — splits a flat list into nested sublists of at most `chunk_size` elements.
* **`add_entries_batched(ids, documents, metadatas)`** — batches and inserts documents with a `tqdm` progress display.

**Ingestion CLI**

```bash
python -m src.mcp_server.retrieval.vector_store \
  --filename "./data/historical_records.parquet" \
  --model_name "./models/fine_tuned_mnrl_checkpoint" \
  --collection_name "coicop_historical_v1" \
  --text_column "product_description" \
  --label_column "coicop_code"
```

| Flag | Full Identifier | Type | Purpose |
| --- | --- | --- | --- |
| `-f` | `--filename` | `str` | Path to `.parquet` or `.csv` source file. |
| `-m` | `--model_name` | `str` | Local path or HF model identifier for embedding. |
| `-c` | `--collection_name` | `str` | ChromaDB collection name to create/populate. |
| `-tc` | `--text_column` | `str` | Column containing text to embed. |
| `-lc` | `--label_column` | `str` | Column containing classification codes. |

---

## Description Generation Pipeline

`DescriptionGenerationPipeline` automates the creation of `detailed_description` fields for a classification system using an LLM. It processes the hierarchy **bottom-up** (from the deepest leaf nodes toward the root), using each node's children as context to generate its parent's description.

**Supported Classification Systems**

| Name | Source | Loader |
| --- | --- | --- |
| `SEA` | XML from Klass server | `XMLDataLoader` |
| `COICOP` | XLSX from UN Statistics | `CoicopDataLoader` |

**Workflow**

```text
Load Classification Data → Initialize ClassificationSystem → LabelDescriptionGenerator
    ↓
For each depth level (max_depth → 1):
    For each code at that level:
        Build prompt (parent + children JSON context)
        Call LLM with exponential backoff (up to 5 retries)
        Store generated description in _lookup
    ↓
Save final JSON to output_path
```

**CLI Usage**

```bash
python -m src.mcp_server.classification_system.description_generation.DescriptionGenerationPipeline \
  --classification-name SEA \
  --path-classification-data ./data/sea_classification.xml \
  --api-key YOUR_KEY \
  --api-base https://api.openai.com/v1 \
  --model-name gpt-4o \
  --max-depth 5 \
  --output-path ./data/sea_with_descriptions.json
```

---

## Model Fine-Tuning Pipeline (`MNRL.py`)

This module fine-tunes a `SentenceTransformer` model for domain-specific retrieval using **Multiple Negatives Ranking Loss (MNRL)**. It is designed for heavily abbreviated product names or ambiguous titles where a general-purpose embedding model underperforms.

**Key Features**

* **Implicit In-Batch Negatives:** For a batch of (anchor, positive) pairs, all positives $P_j$ where $j \neq i$ serve as negatives for anchor $A_i$ — no explicit negative examples required.
* **Balanced Dataset Generation:** Uses `IterableDataset` with a custom `balanced_generator` to keep memory footprint low while preventing class imbalance in batches.
* **MLflow Logging:** Patches the default `MLflowCallback` to sanitize metric keys containing `@` (e.g., `recall@10` → `recall_at_10`), ensuring compatibility with the MLflow backend.

> **Dependency Constraint:** Requires `transformers==4.57.6`. Versions `>=5.x.x` introduce breaking changes that prevent correct loss convergence.

**Process Workflow**

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

**Command Line Arguments**

| Parameter | Shorthand | Type | Function |
| --- | --- | --- | --- |
| `--path_training_data_raw` | `-t` | `str` | Path to the raw `.parquet` dataset. |
| `--path_training_data_storage` | `-s` | `str` | Directory to save train/test split files. |
| `--output_dir` | `-o` | `str` | Directory for fine-tuned weights and checkpoints. |
| `--model_path` | `-m` | `str` | Local path or HF model identifier (e.g., `BAAI/bge-small-en-v1.5`). |
| `--batch_size` | `-b` | `int` | Training batch size per device. |
| `--text_column` | `-tc` | `str` | Column containing text/product descriptions. |
| `--label_column` | `-lc` | `str` | Column containing classification codes. |

**Hyperparameter Configuration**

* **Learning Rate:** `2e-5`
* **Max Steps:** `7500`
* **Evaluation Interval:** every `2500` steps using an IR evaluator on the held-out test set
* **Logging Interval:** every `100` steps to MLflow

**Execution Example**

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

---

## Agents

Two agent implementations are available, suited to different deployment scenarios.

**`agent.py` — MCP Client Agent**

A DSPy `ReAct` agent that connects to the running MCP server over SSE and consumes all tools remotely. This is the intended production setup.

```bash
python agent.py "Ritter Sport Alpenmilch Schokolade"
```

The agent follows a strict SOP defined in `QuestionAnswer`:
1. Analyze the input.
2. Run `full_text_search` (exact noun) or `semantic_search` (complex phrase).
3. Verify top candidate codes using `get_code_specification`.
4. Drill down via `get_children` or abstract via `get_parent` as needed.
5. Output the final SEA code with justification.

**`src/mcp_server/agents/online_retail_classification.py` — Standalone Retail Agent**

A self-contained DSPy `ReAct` agent that calls classification tools directly (without an MCP server). Designed for online retail product classification. Compared to the earlier version in `src/mcp_server/online_retail_classification.py`, it adds:

* A `brand` input field for manufacturer context.
* An `exploration_summary` output field — a required log of all hierarchy nodes visited and the reasoning behind each navigation step.
* `max_tokens=10000` on the underlying LM call.

```python
classification_agent = RetailClassificationAgent()
answer = classification_agent.agent(
    product_name="Bunte Vielfalt, Alpenmilch",
    price="1,99 €",
    brand="Ritter Sport",
    details="Produktdetails\nJe 100-g-Packung\n",
    retailer_category="Süßigkeiten & salzige Snacks  Schokolade"
)
print(answer.exploration_summary)
print(answer.sea)
```

---

## Testing

Unit tests are located in `src/tests/unit/` and use `pytest`. The `pyproject.toml` sets `src/mcp_server` as the Python path so internal imports resolve correctly.

```bash
pytest src/tests/
```

**`test_classification_system.py`** covers:

* **Label normalization** — verifies that codes formatted with dots, spaces, or no separators all normalize identically.
* **`get_code`** — confirms correct retrieval across all formatting variants.
* **Parent–child relationships** — asserts correct child lists for all parent codes.
* **Code traces** — validates full lineage output for all codes and formatting styles.
* **`MarkdownExample` formatting** — unit tests for `header_plus_content`, `generate_examples_part`, and `format_traces_to_markdown`.

Tests use a `system_variant` fixture parameterized over `clean`, `dots`, and `spaces` code formats, so every test runs three times automatically.

---

## Setup and Usage

### Data Requirements

The MCP Server relies on two central data sources:
* **Classification system documentation** — a JSON file in the format described above.
* **Vector database with a fine-tuned embedding model** — if you have access to a reasonably sized set of high-quality annotated examples, run the model training pipeline and embed your historical cases using the provided CLI interfaces.

**Intended setup workflow:**

```text
┌─────────────────┐     ┌──────────────────────────┐     ┌─────────────────────────┐     ┌──────────────────────┐
│   Train Model   │ ──> │ Embed Historic Examples  │ ──> │ Specify Details in .env │ ──> │     Start Server     │
└─────────────────┘     └──────────────────────────┘     └─────────────────────────┘     └──────────────────────┘
```

### Technical Prerequisites

* Python 3.13.2
* A running instance of ChromaDB or a configured local persistent path.
* `.env` file in the root directory with the variables below.

### Environment Variables (`.env`)

```env
# MCP Server / ChromaDB Config
CHROMA_COLLECTION_NAME=your_collection_name
CHROMA_MODEL_NAME=your_embedding_model_name
CHROMA_PATH_CLASSIFICATION_SYSTEM=path/to/sea_classification.json
CHROMA_CLASSIFICATION_NAME=SEA
CHROMA_LABEL_KEY_IN_COLLECTION=coicop
CHROMA_CLIENT_PATH=path/to/chromadb

# StringMatcher / SQLite Config
PATH_TO_DF=path/to/historical_data.parquet
PATH_SQLITE=path/to/fts.db
TEXT_COLUMN=klartext
TABLE_NAME=your_table_name

# Agent Config
SERVER_URL_=http://localhost:8080/sse
MODEL_NAME=your_llm_model  # e.g., openai/gpt-4o
API_BASE=your_api_base
API_KEY=your_api_key

# Model Training / MLflow Config
ML_FLOW_URI=http://127.0.0.1:5000
MODEL_FINETUNING_EXPERIMENT=Retrieval_Model_Training
```

### Running the MCP Server

Start the FastMCP server, which listens for SSE connections on port 8080:

```bash
python src/mcp_server/server.py
```

### Running the MCP Client Agent

```bash
python agent.py "your product or expense description here"
```

---

# Author

Adrian Montag (adrian.montag@destatis.de)