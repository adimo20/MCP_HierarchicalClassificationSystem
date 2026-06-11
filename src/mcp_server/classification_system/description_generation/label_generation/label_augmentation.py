import json
import copy
from typing import Optional, List
from openai import OpenAI, OpenAIError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

class LabelDescriptionGenerator:
    """
    A generator class to synthesize hierarchical label descriptions.

    This class processes classification labels by traversing them bottom-up (level 6 to level 1).
    It leverages an LLM to generate professional, accurate descriptions for parent categories 
    by analyzing the context of their underlying children codes.

    Attributes:
        classification: An instance of the ClassificationSystem containing the hierarchy.
        client: The configured OpenAI-compatible API client.
        model_name: The string identifier of the LLM to use.
        labels: A preprocessed list of classification codes.
    """

    def __init__(self, classification, api_key: str, api_base: str, model_name: str):
        """
        Initializes the generator with classification data and API credentials.

        Args:
            classification: The classification system object.
            api_key: The API key for LLM authentication.
            api_base: The base URL for the API endpoint.
            model_name: The name/version of the model to query.
        """
        self.classification = classification
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name
        self.labels = [
            self.classification._preprocess_label(label.code)
            for label in copy.deepcopy(self.classification.codes)
        ]

    def _get_llm_response(self, prompt: str) -> Optional[str]:
        """
        Sends a prompt to the LLM and retrieves the generated content.

        Implements exponential backoff to handle transient API errors.

        Args:
            prompt: The string containing the instructions and context for the LLM.

        Returns:
            The generated string from the model, or None if the request fails 
            after maximum retries.
        """
        
        @retry(
            wait=wait_random_exponential(min=1, max=60),
            stop=stop_after_attempt(5),
            retry=retry_if_exception_type(OpenAIError),
            reraise=True
        )
        def _call_api():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        try:
            return _call_api()
        except OpenAIError as e:
            print(f"API failure: {e}")
            return None

    def _build_prompt(self, parent_code: str) -> str:
        """
        Constructs a structured prompt for the LLM using parent and child context.

        Args:
            parent_code: The specific code currently being processed.

        Returns:
            A formatted string prompt containing JSON context for children and parent details.
        """
        parent_data = self.classification.get_code(parent_code).to_dict()
        children_data = [c.to_dict() for c in self.classification.get_children(parent_code)]
        
        context = (
            f"# Parent-Code\n```json\n{json.dumps(parent_data, indent=4, ensure_ascii=False)}\n```\n"
            f"# Children\n```json\n{json.dumps(children_data, indent=4, ensure_ascii=False)}\n```"
        )

        return f"""
        You are an expert in official statistics. Your task is to fill in the `detailed_description` field for a parent category accurately, comprehensively, and factually. You are provided with the code and short description of the parent category, as well as a JSON-like array containing all subordinate categories (children).

        Follow these criteria when creating the description:

        1. **Summary of Scope:** Define in 1-2 introductory sentences what this parent category encompasses as a whole, based on the child categories.
        2. **Detailed Inclusions:** Group the most important `keywords` and `explicit_inclusions` of all child categories into logical clusters. Use clear, official statistical language (e.g., "This includes, among others..."). Always cite the explicit subcategories in parentheses.
        3. **Exclusions:** Examine the `exclusions` field for all children. If exclusions are present, state them explicitly and clearly at the end of the description (e.g., "This does not include...").
        4. **Tone & Language:** The text must be written in the language of the input and correspond to a natural, professional tone.

        Here is the input data for the current category:
        {context}

        Please generate exclusively the text for the `detailed_description` of the parent category.
        """

    def generate_descriptions(self, max_depth: int, output_path: str) -> None:
        """
        Orchestrates the bottom-up generation process.

        Iterates through hierarchy levels in reverse, generates descriptions 
        via LLM, and updates the classification system lookup.

        Args:
            max_depth: The deepest level of the hierarchy to process.
            output_path: The file path where the final JSON results will be saved.
        """
        
        for depth in range(max_depth, 0, -1):
            level_codes = [lbl for lbl in self.labels if len(lbl) - 1 == depth]
            
            for code in level_codes:
                try:

                    print(f"Processing: {code}")
                    prompt = self._build_prompt(code)
                    description = self._get_llm_response(prompt)
                    
                    if description:
                        self.classification._lookup[code].detailled_description = description
                except Exception as e:
                    print(f"Error: {e}")
                    continue


        self._save_results(output_path)

    def _save_results(self, output_path: str) -> None:
        """
        Serializes the updated classification data to a JSON file.

        Args:
            output_path: The target file path.
        """
        final_data = [node.to_dict() for node in self.classification._lookup.values()]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)