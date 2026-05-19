import argparse
from typing import Dict, Any, List
from tqdm import tqdm
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()


@register_embedding_function
class CustomEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model:str):
        self.model = SentenceTransformer(model)

    def __call__(self, input_docs: Documents) -> Embeddings:
        # embed the documents somehow
        embeddings = self.model.encode(input_docs)
        return embeddings

    @staticmethod
    def name() -> str:
        return "custom-embedding-function"

    def get_config(self) -> Dict[str, Any]:
        return dict(model=self.model)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        return CustomEmbeddingFunction(config['model'])

class VectorStore:
    def __init__(
        self, 
        collection_name:str, 
        model_name:str, 
        chromadb_path:str
    ) -> None:

        """
        vector store conntect/loads persistent directory either locally or conntects to http chroma client. http client seems to be faster.
        """
        
        self.model_name:str = model_name
        self.embedding_function = CustomEmbeddingFunction(self.model_name)
        print("Loaded embedding model")

        self.chromadb_path:str = chromadb_path
        self.collection_name:str = collection_name
        self.chroma_client = chromadb.PersistentClient(path=self.chromadb_path)
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        print("Loaded chroma collection")


    def chunk_list(
        self,
        list_to_chunk:List,
        chunk_size:int
    )-> List[Any]:
        """
        Chunks a list into chunks of size chunks_size
        """
        return[list_to_chunk[i:i + chunk_size] for i in range(0, len(list_to_chunk), chunk_size)]

    def add_entries_batched(
        self,
        ids:List[str],
        documents:List[str],
        metadatas:List[Any]
    )->None:
        """
        Adds entries to the chroma collection
        """ 
        BATCH_SIZE_CHROMA:int = 5000
        
        documents_chunked:List[str] = self.chunk_list(
            documents, chunk_size=BATCH_SIZE_CHROMA
        )
        
        ids_chunked:List[str] = self.chunk_list(
            ids, chunk_size=BATCH_SIZE_CHROMA
        )
        metadatas_chunked:List[Any] = self.chunk_list(
            metadatas, chunk_size=BATCH_SIZE_CHROMA
        )
        total_steps: int = len(ids_chunked)

        for i, m, d in tqdm(zip(ids_chunked, metadatas_chunked, documents_chunked),total=total_steps):
            self.collection.add(
                ids=i, documents=d, metadatas=m
            )
        return
        
if __name__ == "__main__":

    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add("-f", "--filename",type=str)
    parser.add("-m", "--model_name", type=str)
    parser.add("-c", "collection_name", type=str)
    parser.add("-tc", "--text_column", type=str)
    parser.add("-lc", "--label_column", type=str)

    def fill_vector_store(filename:str, model_name:str, collection_name:str, text_column:str, label_column:str):
        """
        Inserts a dataframe into a chromadb vector database.

        Args:
            filename:str - File to a .parquet dataframe containing the data to be stored
            model_name:str - path to the embedding model which should be used
            collection_name:str - name of the collection we want to store the data in
            text_column:str - column name of the documents we want to embedd in the vector database
            label_column:str - column name of the labels
        Returns:
            None - Stores the embedded documents on an persistent client on disk

        """

        store = VectorStore(
            model_name=model_name,
            collection_name=collection_name
        )
        
        if ".parquet" in filename:
            training_df = pd.read_parquet(filename)
        elif ".csv" in filename:
            try: 
                training_df = pd.read_csv(filename)
            except:  # noqa: E722
                print("Error while reading csv-file. Specify how to read the csv file!")
        else: 
            raise ValueError("Data type not supportet, add custom logic to load your dataset")

        ids: List[str] = [
            f"id_{code}_{i}" for i, code in enumerate(training_df[label_column].to_list())
        ]

        documents:List[str] = training_df[text_column] .to_list()
        __temp_meta_dict: dict = training_df.drop(labels= [text_column],axis=1).reset_index(drop=True).to_dict(orient="index")
        
        metadatas:List[Any] = [
            __temp_meta_dict[i] for i in range(len(__temp_meta_dict))
        ] 
        
        assert len(ids) == len(documents) == len(metadatas)
        
        store.add_entries_batched(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        return
    
    args = parser.parse_args()

    fill_vector_store(
        filename=args.filename,
        model_name=args.model_name,
        collection_name=args.collection_name,
        text_column=args.text_column,
        label_column=args.label_column
    )