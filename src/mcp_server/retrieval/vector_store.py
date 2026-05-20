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
    """
    Registers a custom embedding function that will be used to a) embedd the documents/historical examples inside 
    chromaDB and b) encode queries when retrieving examples from the chromaDB
    The structure of this class follows the chromaDB examples inside the documentation of chroma, for reference look there.
    Args: 
        model (str): either path to the custom trained model on your disk or identified for an embedding model, that will be pulled from hugging face

    """
    def __init__(self, model:str):#Change the name of this argument, because model != self.model 
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

    """
    Interface for filling and querying the chromaDB (persistent) client

        collection_name (str) - name of the collection you want to create/load
        model_name (str) - name of the model to use for embedding and querying the collection
        chromadb_path (str) - Path to the persistent client loaded or to create
    
    """

    def __init__(
        self, 
        collection_name:str, 
        model_name:str, 
        chromadb_path:str
    ) -> None:

        """
        Embedding Model, Chroma Client and Collection is loaded.
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
        list_to_chunk:List[Any],
        chunk_size:int
    )-> List[Any]:
        """
        Splits a list of elements into chunks of size chunk_size, used for adding elements to the chromaDB without 
        breaking the batch size of chroma
        
        Args:
            list_to_chunk (List[Any]) - list to chunk
            chunk_size (int) - maximum chunk size
        """
        return[list_to_chunk[i:i + chunk_size] for i in range(0, len(list_to_chunk), chunk_size)]

    def add_entries_batched(
        self,
        ids:List[str],
        documents:List[str],
        metadatas:List[Any]
    )->None:
        """
        Adds entries to the chroma collection. The lists will be chunked to not break the chromaDB batch size for 
        adding documents.
        Args:
            ids (List[str]) - ids of documents
            documents:List[str] - list of documents/historic examples to embedd
            metadatas:List[Any] - metadata to store alongside the documents, should be list of dictionary, where at least one key value pair corresponds to {"label_or_code":"Label/Code related to a document"}
        Returns:
            None - embedding and documents are stored in the persistent client at the path specified at the beginning on disk
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