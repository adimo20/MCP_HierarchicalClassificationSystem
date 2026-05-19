import re
from retrieval.vector_store import VectorStore
from classification_system.MarkdownAugmentation import MarkdownReport
from collections import defaultdict
from typing import Literal, Dict, Any, List

class Retriever:
        def __init__(
            self, 
            collection_name:str, 
            model_name:str, 
            path_classification_system:str, 
            classification_name:str, 
            label_key_in_collection:str, 
            chromadb_path:str
        ) -> None:
            
            self.model_name = model_name
            self.collection_name = collection_name
            self.classification_name = classification_name
            self.path_classification_system = path_classification_system
            self.label_key_in_collection = label_key_in_collection # the key that the label of each document is stored under in the metadatas
            self.chromadb_path = chromadb_path
            
            self.vector_store = VectorStore(
                collection_name=self.collection_name,
                model_name=self.model_name,
                chromadb_path=self.chromadb_path
            )
            self.load_classification_system()

        def load_classification_system(self) -> None:
            """Initializes the markdown report classification system."""            
            self.classification_system = MarkdownReport(
                path = self.path_classification_system,
                classification_name=self.classification_name
            )

        @staticmethod
        def _clean_code(code: str) -> str:
            """Removes trailing zeros from the classification code."""
            return re.sub(r"0+$", "", code)

        def get_codes_from_retrieved_content(
            self, 
            retrieved_content: Dict[str, Any], 
            label_key = "coicop"
        ) -> List[str]:
            """
            In case you change the spot of the labels in the chroma db just change this!
            """
            return [m.get(label_key) for m in retrieved_content.get("metadatas")]

        def search_collection(
            self,
            q:str,
            k:int,
            search_type:Literal["sim_search", "text_search"]
        ) -> Dict[str, Any]:
            
            """
            Searches the chroma db. Depending on search type a different approach to find relevant cases is used.
            Either `sim_search` performs a similiarity search over the embedded documents or when `text_search` is
            selected a text based search is used, where the condition `text in document == True` will lead a document
            beeing returned.
            """
            query_upper_case: str = q.upper()
            if search_type == "sim_search":
                search_result = self.vector_store.collection.query(
                    query_texts=[query_upper_case], n_results=k
                )
                # Safely handle the nested lists returned by ChromaDB queries
                return {
                    "documents": search_result.get("documents", [[]])[0],
                    "metadatas": search_result.get("metadatas", [[]])[0],
                    "ids": search_result.get("ids", [[]])[0],
                }
                
                
            elif search_type == "text_search":
                return self.vector_store.collection.get(
                    where_document={"$contains":q.upper()}, limit=k, offset=0
                )
            # In case wrong search type was used
            raise ValueError(f"Unsupported search_type: {search_type}")

        def get_unique_codes(
            self,
            q:str,
            k:int,
            label_key:str="coicop",
            search_type:Literal["sim_search", "text_search"]="sim_search"
        )->List[str]:
            """
            Query the vector store to get the k most related entries and returns the unique codes
            """
            retrieved_content = self.search_collection(
                q=q, k=k, search_type=search_type
            )
            codes: List[str] = self.get_codes_from_retrieved_content(
                retrieved_content=retrieved_content, label_key=label_key
            )
            unique_codes = []
            
            unique_codes: List[str] = list(dict.fromkeys(filter(None, codes)))

            code_example_dict =  defaultdict(list)
            
            for code, entry in zip(codes, retrieved_content.get("documents", [])):
                if code:
                    clean_code: str = self._clean_code(code)
                    code_example_dict[clean_code].append(entry)

            return unique_codes, code_example_dict

        def create_augmented_context(
            self, q:str,k:int, use_examples:bool=True, search_type:Literal["sim_search", "text_search"]="sim_search"
        )->str:
            
            """
            Retrieves relevant classification data and builds an augmented context string.

            This method performs a semantic search via the vector store to find the 
            top 'k' related historical entries for a given input query. It extracts 
            unique classification codes, cleans them (removing trailing zeros), and 
            delegates to the classification system to generate a detailed Markdown 
            summary. This output is designed to serve as "context" for a RAG 
            (Retrieval-Augmented Generation) prompt.

            Args:
                q (str): The raw input product description or query string 
                    (e.g., "Organic Whole Grain Wheat Bread").
                k (int): The number of nearest neighbor results to retrieve 
                    from the vector store before filtering for unique codes.

            Returns:
                str: A comprehensive Markdown-formatted string containing the original 
                    input, relevant category descriptions, and their hierarchical 
                    classification paths.

            Note:
                The method automatically handles code normalization by stripping 
                trailing zeros, ensuring that codes like '011100' are correctly 
                mapped to their functional tree counterparts like '0111'.
            """
            
            unique_codes, code_example_dict = self.get_unique_codes(
                q=q,
                k=k,
                label_key=self.label_key_in_collection,
                search_type=search_type
            )

            if not use_examples:
                code_example_dict = None

            cleaned_unique_codes: List[str] = [self._clean_code(code) for code in unique_codes]
            markdown_summary: str = self.classification_system.generate_markdown_summary(cleaned_unique_codes, examples_dict=code_example_dict)
            if markdown_summary is None or markdown_summary == "":
                markdown_summary = "## Search Result \n**No Products** that match the input product could be identified."
            return f"# Input \n **Input-Product**: {q} \n\n{markdown_summary}"


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()

    retriever = Retriever(
        collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
        model_name=os.getenv("CHROMA_MODEL_NAME"),
        chromadb_path = os.getenv("CHROMA_CLIENT_PATH"),
        path_classification_system=os.getenv("CHROMA_PATH_CLASSIFICATION_SYSTEM"),
        classification_name=os.getenv("CHROMA_CLASSIFICATION_NAME"),
        label_key_in_collection=os.getenv("CHROMA_LABEL_KEY_IN_COLLECTION")
    )
    
    test_result = retriever.create_augmented_context(
        q="Adidas Speziale",
        k=25,
        use_examples=True,
        search_type="sim_search"
    )
    print(test_result)



        
        
