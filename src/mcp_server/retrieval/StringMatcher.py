import pandas as pd
import sqlite3
from typing import Any
from time import process_time
import re

class StringMatcher:
    def __init__(
        self,
        path_to_df: str,
        path_sqlite: str,
        text_column: str,
        label_column: str,
        table_name: str
    ) -> None:

        self.path_to_df = path_to_df
        self.path_sqlite = path_sqlite
        self.text_column = text_column
        self.label_column = label_column
        self.table_name = table_name
        self.conn = sqlite3.connect(self.path_sqlite)
        self._initialize_db()

    def _preprocess_label(
        self, 
        label:str
    )->str:
        """Normalises a label according to the way it is normalised in the classification system"""
        return re.sub(r"0+$", "", re.sub(r"[^0-9A-Za-z]", "", label))

    def _initialize_db(self) -> None:
        """Loads data from a source file and initializes the SQLite database.

        Reads a source file (CSV or Parquet) into a pandas DataFrame, creates a
        temporary standard SQLite table, and then performs a migration to an
        FTS5 virtual table for optimized full-text searching.

        Raises:
            ValueError: If the file extension is not supported (.csv or .parquet).
            sqlite3.Error: If there is an issue during database initialization,
                table creation, or data insertion.
        """
        s = process_time()
        print("Initialising sqlite database")
        
        if self.path_to_df.endswith(".csv"):
            df = pd.read_csv(self.path_to_df)
        elif self.path_to_df.endswith(".parquet"):
            df = pd.read_parquet(self.path_to_df)
        else:
            raise ValueError("Data type not supported! Use .csv or .parquet.")

        df[self.label_column] = df[self.label_column].apply(lambda s: self._preprocess_label(s))
        temp_table = f"{self.table_name}_temp"
        df.to_sql(temp_table, self.conn, if_exists='replace', index=False)

        cursor = self.conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        cursor.execute(f"""
            CREATE VIRTUAL TABLE {self.table_name} USING fts5(
                {self.label_column}, 
                {self.text_column}
            )
        """)
        
        cursor.execute(f"""
            INSERT INTO {self.table_name} 
            SELECT {self.label_column}, {self.text_column} FROM {temp_table}
        """)

        cursor.execute(f"DROP TABLE {temp_table}")
        self.conn.commit()
        print(f"Process time for data base intialisation: {process_time() - s}")
        

    def query_db(
        self,
        query: str
    ) -> list[tuple[Any]]:
        cursor = self.conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    
    def organise_data(
        self, 
        query_results:list[tuple[str, str]],
        num_examples_cap:int=10
    ) -> tuple[list[str], dict[str,list[str]]]:

        df = pd.DataFrame(query_results, columns=["labels", "texts"])
        df['texts'] = df['texts'].apply(lambda x: "".join(x) if isinstance(x, (list, tuple)) else str(x))
        result_dict = (
            df.groupby("labels")["texts"]
            .apply(lambda x: list(x)[:num_examples_cap])
            .to_dict()
        )
        unique_labels = list(result_dict.keys())
        return unique_labels, result_dict

    def match_data(
        self,
        q:str,
        k_per_class:int
    )->dict[str,list[str]]|tuple[None,None]:

        s = process_time()
        
        query_results = self.query_db(
            f"""
            SELECT {self.label_column}, {self.text_column}
            FROM {self.table_name} 
            WHERE klartext='{q.upper()}'
            """
        )

        if query_results == []: 
            print("Executing string in string")
            query_results = self.query_db(
            f"""
            SELECT {self.label_column}, {self.text_column}
            FROM {self.table_name} 
            WHERE klartext MATCH '{q.upper()}'
            """
            )
        print(f"Process time: {process_time() - s}")
        if query_results:
            return self.organise_data(
                query_results,
                num_examples_cap=k_per_class
            )
        return None, None  

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
