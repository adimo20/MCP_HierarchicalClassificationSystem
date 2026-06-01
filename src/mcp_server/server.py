from fastmcp import FastMCP
from dotenv import load_dotenv
from retrieval.vector_store import VectorStore
from retrieval.Retriever import Retriever
from retrieval.StringMatcher import StringMatcher
from classification_system.MarkdownAugmentation import MarkdownReport

from typing import Any
import os
import json

load_dotenv()

vector_store = VectorStore(
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
    vector_store=vector_store,
    string_matcher=matcher,
    classification_system=classification_system
)

del(matcher, classification_system, vector_store)

mcp: FastMCP[Any] = FastMCP(
    name="MCP-Server for the Classification of products according to the SEA (Systematik der Einnahmen und Ausgaben der Privaten Haushalte)"
)

@mcp.tool
def get_root_category_codes_and_descriptions() -> list[dict]:
    """
    Returns the top-level divisions (root categories) of the SEA classification system.
    
    WHEN TO USE THIS:
    - Use this as your starting point when you have a completely new item to classify and you don't know which general category it belongs to (e.g., Food vs. Clothing vs. Transport).
    - Use this to understand the highest-level structure of the SEA classification.
    
    Args:
        None: no input required.
        
    Returns:
        list[dict]: A list of dictionaries containing the top-level 'code' (e.g., '01', '02') and its overarching 'description'.
    """
    root_categories: list[str] = [f"0{i}" if i < 10 else str(i) for i in range(16)]
    return [
        {
            "code": retriever.classification_system.classification.get_code(code).to_dict().get("code"),
            "description": retriever.classification_system.classification.get_code(code).to_dict().get("description"),
        } for code in root_categories
    ]

@mcp.tool
def get_children(parent_code: str) -> list[dict[str,str]]:
    """
    Collects a list of direct child categories for a given parent code within the SEA system.
    
    WHEN TO USE THIS:
    - Use this to drill down hierarchically into the classification tree. 
    - Once you have identified a broad category (e.g., '01' for Food), use this tool to find the next level of specificity (e.g., '011' for Bread and Cereals).
    - Repeat this process until you reach the lowest level (leaf node) that accurately describes the product.
    
    Args:
        parent_code (str): The classification code you want to explore the children of (e.g., '01' or '011'). Do not include trailing zeros.
        
    Returns:
        list[str]: A list of JSON strings detailing the child categories, their codes, and descriptions.
    """
    children: list = retriever.classification_system.classification.get_children(
        parent=parent_code
    )
    children_json: dict[str,str] = [
        {
            "code":c.code,
            "description":c.description
        } for c in children
    ]
    return children_json

@mcp.tool
def get_parent(parent_code: str) ->str:
    """
    Retrieves the immediate parent category for a given overly specific SEA code.
    
    WHEN TO USE THIS:
    - Use this for ABSTRACTION. When your search finds a highly specific SEA code, 
      but the user's input lacks the necessary details to justify that level of 
      granularity, you must move one level up the hierarchy.
    - Example 1 (Missing Demographic): You found the code `03121` (Bekleidung für Herren), 
      but the user's expense description is simply "T-Shirt" and does not specify if it 
      was for a man, woman, or child. Use this tool to back up to the broader parent 
      code `0312` (Bekleidung).
    - Example 2 (Missing Specification): You found `01141` (Roh- und Vollmilch), but 
      the input just says "Milch". Use this tool to fall back to the safest shared 
      parent `0114` (Milch, andere Molkereiprodukte und Eier).
    
    Args:
        specific_code (str): The overly specific classification code you want to abstract 
                             upwards from (e.g., '03121' or '01141'). Do not include trailing zeros.
        
    Returns:
        str: A JSON string detailing the broader parent category, its code, and description.
    """
    len_parent_code: int = len(parent_code)
    child_code:str = parent_code[:len_parent_code-1]
    parent = retriever.classification_system.classification.get_code(
        code=child_code
    )

    children_json:str = json.dumps({
        "code":parent.code,
        "description":parent.description
    }, indent=4, ensure_ascii=False)

    return children_json


@mcp.tool
def get_code_specification(list_of_codes: list[str]) -> str:
    """
    Generates a comprehensive, definitive Markdown report for specific SEA classification codes.
    
    WHEN TO USE THIS:
    - Use this only with codes **that you do not already have seen a detailled descritpiton** when using a semantic or fulltext search.
    - Use this to VERIFY if a product belongs in a specific category.
    - Use this when you need the official rules, inclusions, and exclusions for a specific code.
    - If you are debating between two or more codes, pass them both in the list to compare their exact specifications.
    
    This method retrieves the exact hierarchical trace (path) through the classification system, 
    and detailed descriptive texts to help make a final classification decision and understand the semantic meaning of a code.

    Args:
        list_of_codes (list[str]): A list of code strings to retrieve detailed information for. 
            Must always be a list, even for a single code (e.g., ['01111'] or ['01111', '01211']).

    Returns:
        str: A formatted Markdown string containing comprehensive summaries for all requested valid codes, 
             separated by horizontal rules (---).
    """
    return retriever.classification_system.generate_markdown_summary(
        list_of_codes=list_of_codes
    )

@mcp.tool
def semantic_search(q: str, k: int) -> str:
    """
    Performs a natural language similarity search over historical classified products to suggest the best SEA codes.
    
    WHEN TO USE THIS:
    - Use this when you have a complex, descriptive, or unusual product string (e.g., "Bio Vollkornbrot mit Sonnenblumenkernen" or "Organic Whole Grain Wheat Bread").
    - Use this when exact keyword searches fail or return no results.
    - It leverages vector embeddings to find *conceptually* similar past classifications.
    
    This acts as a RAG (Retrieval-Augmented Generation) context builder. It finds the top 'k' 
    related historical entries, extracts their codes, normalizes them, and returns a detailed 
    Markdown summary of those suggested categories.

    Args:
        q (str): The raw, natural language product description (e.g., "Organic Whole Grain Bread").
        k (int): The number of nearest neighbor results to retrieve from the vector database. 
                 A good default is 10 to 20 depending on desired broadness.

    Returns:
        str: A comprehensive Markdown-formatted context string containing the original input, 
             suggested category descriptions, their hierarchical paths, and historic example products.
    """
    return retriever.create_augmented_context(
        q=q,
        k=k,
        use_examples=True,
        search_type="sim_search"
    )
            
@mcp.tool
def full_text_search(q: str, k: int) -> str:
    """
    Performs an exact keyword/substring search over historical classified products to suggest SEA codes.
    
    WHEN TO USE THIS:
    - Use this for short, exact nouns or standardized product names (e.g., "Hose", "Milch", "Jeans").
    - Use this when you want to see exactly how a specific word was classified in the past.
    - Use this if 'semantic_search' returns results that are conceptually similar but factually wrong.
    
    This method performs a raw text search (`text in document`) via the historic examples database.
    It extracts unique classification codes associated with those matches and generates a Markdown summary.

    Args:
        q (str): The exact keyword or substring to search for (e.g., "Hose" or "Bread").
        k (int): The maximum number of matching database results to retrieve before filtering. 
                 A good default is 10 to 20.

    Returns:
        str: A comprehensive Markdown-formatted context string containing the matched categories, 
             their detailed descriptions, and the historic example products that triggered the match.
    """
    return retriever.create_augmented_context(
        q=q,
        k=k,
        use_examples=True,
        search_type="text_search"
    )

if __name__ == "__main__":
    
    mcp.run(transport="sse", port=8080)