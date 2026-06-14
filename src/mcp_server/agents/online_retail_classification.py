
from dotenv import load_dotenv
from classification_system.MarkdownAugmentation import MarkdownReport

from typing import Any
import os
import json
import dspy
import mlflow

load_dotenv()


mlflow.set_tracking_uri(os.getenv("ML_FLOW_URI"))
mlflow.set_experiment(
    experiment_name="test_agent"
)
mlflow.dspy.autolog()


class ClassificationSignature(dspy.Signature):
    """
    You are an expert taxonomist and retail data classification agent.
    Your task is to accurately classify a given retail product into the "Systematik der Einnahmen und Ausgaben Privater Haushalte" (SEA) classification system.

    You will be provided with the product's name, brand, price, descriptive details, and the retailer's internal category.

    You MUST follow this exact iterative navigation procedure. Do not skip, reorder, or shortcut these steps.

    STEP 1 — START AT THE ROOT:
    - Begin every classification by calling `get_root_category_codes_and_descriptions` to retrieve all top-level divisions.

    STEP 2 — SELECT RELEVANT CODES:
    - From the returned list, select the relevant candidate code(s) that could plausibly contain the product, based on the product data (name, brand, details, retailer category).
    - You may select more than one candidate if several are plausible.

    STEP 3 — INSPECT SPECIFICATIONS:
    - Call `get_code_specification` on the relevant code(s) you selected.
    - Read the official rules, inclusions, and exclusions to confirm the product genuinely fits before descending further. Use the specifications to discard candidates that do not fit.

    STEP 4 — GO ONE LEVEL DEEPER:
    - For the confirmed relevant code, call `get_children` to retrieve its direct child categories (this increases the hierarchy level by one).

    STEP 5 — LOOP:
    - Repeat the cycle of [select relevant codes from children] → [`get_code_specification` to verify] → [`get_children` to go one level deeper] for each level of the hierarchy.
    - At every level you must re-select the relevant code(s) from the children, verify them with `get_code_specification`, and only then descend again with `get_children`.

    STEP 6 — TERMINATE:
    - Stop the loop and produce the output ONLY when you have reached and verified a correct SEA level-4 node (a 5-digit code). This 5-digit code is the terminal condition.
    - Do not terminate early on a 2-, 3-, or 4-digit code.

    ABSTRACTION RULE:
    - If descending leads you toward a code that is more specific than the input data justifies (e.g., it requires a demographic, material, or specification that is not present in the product data), use `get_parent` to step back up to the safest broader parent. Never guess missing demographics or materials.

    CRITICAL — EXPLORATION SUMMARY:
    - In `exploration_summary` you MUST log the full path taken: every node visited (code and description) in order, which codes you selected as relevant at each level and why, what the specifications told you, where you went deeper, and any point where you used `get_parent` to abstract back up. Justify the final 5-digit code based on the specifications you read.
    """
    product_name: str = dspy.InputField(desc="Product to classify.")
    brand: str = dspy.InputField(desc="The brand or manufacturer of the product.")
    price: str = dspy.InputField(desc="Price of the product to classify.")
    details: str = dspy.InputField(desc="Details provided by the retailer.")
    retailer_category: str = dspy.InputField(desc="Product category the retailer lists this product as.")

    exploration_summary: str = dspy.OutputField(
        desc="A detailed, ordered log of every hierarchy node visited (code + description), the relevant codes selected at each level, the specification checks performed, the level-deeper steps taken, and the reasoning behind reaching the final 5-digit SEA code."
    )
    sea: str = dspy.OutputField(
        desc="The final 5-digit SEA code that most accurately fits the product. NEVER write a description into this field. Just write the code in this field."
    )

class RetailClassificationAgent:

    def __init__(self):
        
        self.classification_system = MarkdownReport(
            path=os.getenv("CHROMA_PATH_CLASSIFICATION_SYSTEM"),
            classification_name=os.getenv("CHROMA_CLASSIFICATION_NAME"),
        )
        self.__initialise_lm()
        self.agent = dspy.ReAct(
            signature=ClassificationSignature,
            tools=[
                self.get_root_category_codes_and_descriptions,
                self.get_children,
                self.get_parent,
                self.get_code_specification
            ]
        )

    def __initialise_lm(self)->None:
        lm = dspy.LM(
            model=os.getenv("MODEL_NAME"),
            api_key=os.getenv("API_KEY"),
            max_tokens = 10000
        )
        dspy.configure(lm=lm)


    def get_root_category_codes_and_descriptions(self) -> list[dict]:
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
                "code": self.classification_system.classification.get_code(code).to_dict().get("code"),
                "description": self.classification_system.classification.get_code(code).to_dict().get("description"),
            } for code in root_categories
        ]


    def get_children(self, parent_code: str) -> list[dict[str,str]]:
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
        children: list = self.classification_system.classification.get_children(
            parent=parent_code
        )
        children_json: dict[str,str] = [
            {
                "code":c.code,
                "description":c.description
            } for c in children
        ]
        return children_json


    def get_parent(self, parent_code: str) ->str:
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
        parent = self.classification_system.classification.get_code(
            code=child_code
        )

        children_json:str = json.dumps({
            "code":parent.code,
            "description":parent.description
        }, indent=4, ensure_ascii=False)

        return children_json



    def get_code_specification(self, list_of_codes: list[str]) -> str:
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
        return self.classification_system.generate_markdown_summary(
            list_of_codes=list_of_codes
        )


if __name__ == "__main__":
    classification_agent = RetailClassificationAgent()
    print("Initialised agent!\nClassifying product now!")
    answer = classification_agent.agent(
        product_name=  "MAIZENA Malzextrakte",
        price= "2,19 €",
        brand="MAIZENA",
        details=rf"Reine Malzextrakte",
        retailer_category=" Startseite Vorräte Grundnahrungsmittel Hülsenfrüchte, Mais & Getreide MAIZENA Malzextrakte" 
    )
    print(answer.reasoning)
    print(answer.sea)

