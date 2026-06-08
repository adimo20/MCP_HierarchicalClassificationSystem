import re
from retrieval.vector_store import VectorStore
from retrieval.StringMatcher import StringMatcher
from classification_system.MarkdownAugmentation import MarkdownReport
from collections import defaultdict
from typing import Literal, Dict, Any, List

class Retriever:

        """
        Retrieves relevant labels from a chromaDB (containing historic labelled data) and returns a comprehensive markdown
        summary about the retrieved content/examples and their corresponding codes

        Args:
            chromadb_path (str) - path to you chromaDB directory.
            collection_name (str) - Name of the collection where the embedded historical labelled data is stored inside the chroma persistent client
            model_name (str) - Name of the embedding model used to embedded the historical examples IMPORTANT make sure this matches the model you used while embedding the examples otherwise you'll receive non-sense examples when you query the retriever
            path_classification_system (str) - path to the json object where the classification system details are stored.
            classification_name (str) - name of the classification system you use, e.g. COICOP, SEA, NACE
            label_key_in_collection (str) - key under which you stored the labels inside of the metadata inside the chroma collection
            
        """

        def __init__(
            self,  
            label_key_in_collection:str, 
            vector_store:VectorStore,
            string_matcher:StringMatcher,
            classification_system:MarkdownReport
        ) -> None:
            
            
            self.label_key_in_collection = label_key_in_collection # the key that the label of each document is stored under in the metadatas
            
            # Saving Preloaded VectorStore, StringMatcher, MarkdownReport to init via depency injection 
            self.vector_store = vector_store
            self.string_matcher = string_matcher
            self.classification_system = classification_system

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
            Codes/Labels are stored in the metadata of documents inside chormaDB
            Extracts the labels from all the entries in metadata. 
            In case you change the spot of the labels in the chroma db just change this!
            """
            return [m.get(label_key) for m in retrieved_content.get("metadatas")]

        def search_collection(
            self,
            q:str,
            k:int
        ) -> Dict[str, Any]:
            
            """
            Searches the chroma db performs a similiarity search over the embedded documents.
            """
            query_upper_case: str = q.upper()
            
            search_result = self.vector_store.collection.query(
                query_texts=[query_upper_case], n_results=k
            )
            # Safely handle the nested lists returned by ChromaDB queries
            return {
                "documents": search_result.get("documents", [[]])[0],
                "metadatas": search_result.get("metadatas", [[]])[0],
                "ids": search_result.get("ids", [[]])[0],
            }
                

        def get_unique_codes(
            self,
            q:str,
            k:int,
            label_key:str="coicop"
        )->tuple[list[str], dict[str, list[str]]]:
            """
            Query the vector store to get the k most related entries, collects all relevant and unique codes and arranges the entries by label.

            Args:
                q (str) - query term you want to look up in the db
                k (int) - number of examples you want to retrieve
                label_key (str) - key under which the code is stored in the db
            Returns:
                list[str] - containing all the codes/labels retrieved from the vectorDB
                dict[str, list[str]] - containing the labels as keys and a list as values containing all the examples related to the key

            """
            retrieved_content = self.search_collection(
                q=q, k=k
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
            self,
            q:str,
            k:int,
            use_examples:bool=True,
            search_type:Literal["sim_search", "text_search"]="sim_search"
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
            if search_type == "sim_search":
                unique_codes, code_example_dict = self.get_unique_codes(
                    q=q,
                    k=k,
                    label_key=self.label_key_in_collection
                )
            elif search_type == "text_search":
                unique_codes, code_example_dict = self.string_matcher.match_data(
                    q=q,
                    k_per_class=k
                )
                print(code_example_dict)
                if unique_codes is None and code_example_dict is None: return f"# Input \n **Input-Product**: {q} \n\n **No matching item found!**"
            else:
                raise ValueError(f"Unsupported search_type: {search_type}")


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
    import sys

    load_dotenv()

    vs = VectorStore(
        collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
        model_name=os.getenv("CHROMA_MODEL_NAME"),
        chromadb_path=os.getenv("CHROMA_CLIENT_PATH")
    )
    # 
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
    
    test_result = retriever.create_augmented_context(
            q=sys.argv[1],
            k=25,
            use_examples=True,
            search_type="text_search"
        )
    print(test_result)



        
        