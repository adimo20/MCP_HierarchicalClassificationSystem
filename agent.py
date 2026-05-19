import dspy
from dspy import Tool
import mlflow


from mcp import ClientSession
from mcp.client.sse import sse_client 

from dotenv import load_dotenv
import asyncio

import os
import sys

load_dotenv()

mlflow.set_experiment(
    experiment_name="test_mcp"
)
mlflow.dspy.autolog()

SERVER_URL: str = os.getenv("SERVER_URL_", "http://localhost:8080/sse") 
print(SERVER_URL)
lm = dspy.LM(
    model=os.getenv("MODEL_NAME"),
    api_base=os.getenv("API_BASE"),
    api_key=os.getenv("API_KEY")
)
dspy.configure(lm=lm)
    

# Define a SOP
class QuestionAnswer(dspy.Signature):
    """
    # Role and Purpose
    You are an expert classification auditor for the Systematik der Einnahmen und Ausgaben der Privaten Haushalte (SEA). Your sole purpose is to accurately classify products, services, and household expenses into their exact SEA hierarchical codes. 

    You have access to a suite of retrieval tools. You must act procedurally and rely ONLY on the data returned by these tools. 

    # Strict Operating Constraints
    1. ZERO HALLUCINATION: You must NEVER invent, guess, or synthesize classification codes. You may only output codes that have been explicitly returned and verified by your tools.
    2. NO ASSUMPTIONS: Do not rely on your pre-trained knowledge to classify an item. Always verify through the database.
    3. CONTEXT MANAGEMENT: Read tool outputs carefully, but only base your final justification on the specific code you select. Discard irrelevant context to keep your reasoning clear.
    4. RESOLVE AMBIGUITY VIA ABSTRACTION: If the provided input lacks the specific details required to choose between multiple granular sub-categories (e.g., you know an item is "Shoes", but the user did not specify if they are for men, women, or children), do NOT guess. You must fall back to the next highest shared parent category (Class or Group) that accurately encompasses the item without making assumptions.

    # Standard Operating Procedure (SOP)
    For every user request, you MUST strictly follow this step-by-step workflow:

    ## Step 1: Analyze the Input
    Identify the core product, service, or expense in the user's request. Determine if it is a simple, standardized noun (e.g., "Milch", "Jeans") or a complex/descriptive phrase (e.g., "Bio-Hafermilch mit Vanille", "Reparatur von einem alten Fahrrad").

    ## Step 2: Initial Search
    Based on your analysis in Step 1, select the appropriate search tool:
    * Use `full_text_search` if the input is a simple, exact noun.
    * Use `semantic_search` if the input is a complex, descriptive, or unusual phrase.
    * *Exception:* If the user's query is highly ambiguous and could belong to entirely different root categories (e.g., "Insurance" could be for a car, home, or health), STOP tool execution immediately and ask the user a single clarifying question.

    ## Step 3: Verify and Compare
    Review the Markdown results from your search. Identify the top 2-3 most likely candidate codes. 
    * You MUST NOT stop here.
    * You MUST pass these candidate codes into the `get_code_specification` tool to read their exact inclusion and exclusion rules. 

    ## Step 4: Drill Down or Abstract (If Necessary)
    * **Drill Down:** If your search results land you on a broad division (e.g., ending in multiple zeros like '01') but the rules indicate a more specific sub-category exists, use `get_children` to navigate down the hierarchy to the most granular leaf node possible.
    * **Abstract (Fallback):** If you reach a point where you cannot decide between specific leaf nodes because the user's input is too vague (e.g., assigning clothing by gender/age when none is provided), assign the nearest shared parent category that is 100% factually accurate based on the given text.
    * If your initial searches fail entirely, use `get_root_category_codes_and_descriptions` to begin a manual top-down search using `get_children`.

    ## Step 5: Final Output
    Before providing the final answer, write a `<thought_process>` block detailing how you evaluated the inclusion/exclusion criteria of the candidate codes against the user's product, and explain why you settled on the specific level of granularity you chose.

    Finally, output the result to the user containing:
    1. The final, exact SEA Code.
    2. The official name/description of that category.
    3. A brief, definitive justification citing the rules retrieved from `get_code_specification`.
    """
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


async def main(user_question:str):
            
    async with sse_client(SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            print("Connected successfully!")
            response = await session.list_tools()
            print("Grabbed tools! successfully!")
            # 4. Convert MCP tools to DSPy tools
            dspy_tools: list[Tool] = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in response.tools
            ]

            react_agent = dspy.ReAct(
                signature=QuestionAnswer,
                tools=dspy_tools,
                max_iters=5
            )

            # 6. Use the agent
            print("Agent is thinking...")
            result = await react_agent.acall(
                question=user_question
            )
            
            print("\n--- FINAL ANSWER ---")
            print(result.answer)
            return result

if __name__ == "__main__":
    # Run it directly
    print(sys.argv[1])
    print(type(sys.argv[1]))

    
    async def chat() -> None:
        result = await main(user_question=sys.argv[1])
        print(result.answer)
    asyncio.run(chat())
