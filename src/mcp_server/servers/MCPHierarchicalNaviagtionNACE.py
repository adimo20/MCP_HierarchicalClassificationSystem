from fastmcp import FastMCP
from dotenv import load_dotenv
from classification_system.MarkdownAugmentation import MarkdownReport

from typing import Any
import os
import json

load_dotenv()

classification_system = MarkdownReport(
    path=os.getenv("CHROMA_PATH_CLASSIFICATION_SYSTEM"),
    classification_name=os.getenv("CHROMA_CLASSIFICATION_NAME"),
)


mcp: FastMCP[Any] = FastMCP(
    name="MCP-Server for the NACE Classification of Economic Activities"
)


@mcp.tool
def get_root_category_codes_and_descriptions() -> list[dict]:
    """
    Returns the top-level sections (root categories) of the NACE Rev. 2
    Classification of Economic Activities.

    WHEN TO USE THIS:
    - Use this as your starting point when you have a completely new activity
      to classify and you don't know which broad section it belongs to
      (e.g., Agriculture vs. Manufacturing vs. Financial Services).
    - Use this to understand the highest-level structure of the NACE
      classification before drilling down.

    Args:
        None: no input required.

    Returns:
        list[dict]: A list of dictionaries each containing a top-level
            'code' (the single-letter section, e.g., 'A', 'C', 'G') and
            its overarching 'description'
            (e.g., 'Agriculture, Forestry and Fishing',
             'Manufacturing',
             'Wholesale and Retail Trade').
    """
    root_categories: list[str] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V']
    return [
        {
            "code": classification_system.classification.get_code(code).to_dict().get("code"),
            "description": classification_system.classification.get_code(code).to_dict().get("description"),
            "children_categories":classification_system.classification.get_code(code).to_dict().get("details").get("children_codes")
        }
        for code in root_categories
    ]


@mcp.tool
def get_children(parent_code: str) -> list[dict[str, str]]:
    """
    Returns the direct child categories for a given NACE parent code,
    allowing step-by-step navigation down the hierarchy.

    WHEN TO USE THIS:
    - Use this to drill down through the NACE hierarchy level by level:
        Section  →  Division  →  Group  →  Class
        e.g.  'C'  →  '10'  →  '10.1'  →  '10.11'
    - Once you have identified a broad section (e.g., 'C' for Manufacturing),
      call this tool to list its divisions (e.g., '10' Manufacture of Food
      Products, '11' Manufacture of Beverages, …).
    - Repeat until you reach the leaf node (Class level) that most accurately
      describes the economic activity.

    Args:
        parent_code (str): The NACE code whose children you want to explore.
            Use the code exactly as it appears in the hierarchy — sections as
            a single letter (e.g., 'C'), divisions as two digits (e.g., '10'),
            groups with one decimal place (e.g., '10.1').
            Do NOT include trailing zeros.

    Returns:
        list[dict[str, str]]: A list of dicts, each with 'code' and
            'description' for every direct child.
            Example — children of division '62' (Computer programming):
              [
                {"code": "62.0",  "description": "Computer programming, consultancy and related activities"},
                ...
              ]
    """
    children: list = classification_system.classification.get_children(parent=parent_code)
    children_json: dict[str, str] = [
        {"code": c.code, "description": c.description} for c in children
    ]
    return children_json


@mcp.tool
def get_parent(parent_code: str) -> str:
    """
    Retrieves the immediate parent category for an overly specific NACE code,
    moving one level up the hierarchy.

    WHEN TO USE THIS:
    - Use this for ABSTRACTION. When a search returns a very specific NACE
      class, but the input lacks the detail needed to justify that granularity,
      step back to the broader parent.
    - Example 1 (Missing Specialisation): You found class '47.11'
      (Retail sale in non-specialised stores with food, beverages or tobacco
      predominating), but the input is simply "Supermarket" without further
      context. Use this tool to back up to group '47.1'
      (Retail sale in non-specialised stores).
    - Example 2 (Missing Activity Type): You found class '41.10'
      (Development of building projects), but the input only says
      "Construction company". Use this tool to move up to division '41'
      (Construction of buildings).

    Args:
        parent_code (str): The overly specific NACE code you want to abstract
            upwards from (e.g., '47.11' or '41.10').
            Do NOT include trailing zeros.

    Returns:
        str: A JSON string with the broader parent category, e.g.:
            {
                "code": "47.1",
                "description": "Retail sale in non-specialised stores"
            }
    """
    len_parent_code: int = len(parent_code)
    child_code: str = parent_code[:len_parent_code - 1]
    parent = classification_system.classification.get_code(code=child_code)

    children_json: str = json.dumps(
        {"code": parent.code, "description": parent.description},
        indent=4,
        ensure_ascii=False,
    )
    return children_json


@mcp.tool
def get_code_specification(list_of_codes: list[str]) -> str:
    """
    Generates a comprehensive Markdown report for one or more specific NACE
    codes, including the full hierarchical path and official inclusion /
    exclusion rules.

    WHEN TO USE THIS:
    - Use this only for codes whose detailed specification you have NOT yet
      seen (e.g., after a semantic or full-text search returned a candidate
      code but no description text).
    - Use this to VERIFY whether a particular economic activity belongs in a
      candidate category by checking its official inclusions and exclusions.
    - Use this when you need the authoritative definition of a code
      (e.g., to distinguish '69.10' Legal activities from '69.20'
      Accounting, bookkeeping and auditing activities).
    - If you are deciding between two or more codes, pass them all in the
      list to compare their specifications side by side.
      Example: ['46.19', '46.90'] to compare different wholesale trade codes.

    Args:
        list_of_codes (list[str]): A list of NACE code strings to retrieve
            detailed information for. Must always be a list, even for a single
            code.
            Examples:
              Single code  → ['62.01']
              Multiple     → ['62.01', '62.02', '62.09']

    Returns:
        str: A formatted Markdown string containing a comprehensive summary
            for each requested valid code (hierarchical path + descriptive
            text), with individual reports separated by horizontal rules (---).
    """
    return classification_system.generate_markdown_summary(list_of_codes=list_of_codes)


if __name__ == "__main__":
    mcp.run(transport=os.getenv("MCP_TRANSPORT_METHOD"), host="0.0.0.0")