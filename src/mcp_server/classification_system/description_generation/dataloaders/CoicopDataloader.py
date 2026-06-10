import pandas as pd
import re

class CoicopDataLoader:
  def __init__(self, path=None) -> None:
    self.url = "https://unstats.un.org/unsd/classifications/Econ/Download/COICOP_2018_English_structure.xlsx"
    self.path = path

  def load_xlsx_dataset(self, location:str)->pd.DataFrame:
    """
    Downloads the coicop documentation dataset from the official website of the UNSD.
    Parameters:
      url:str
    Returns:
      pd.DataFrame
    """
    coicop_df = pd.read_excel(location).fillna("")
    coicop_df.columns = ["code", "description", "detailled_description", "includes", "alsoIncludes", "excludes"]
    return coicop_df

  def fix_encoding(self, df:pd.DataFrame)->pd.DataFrame:
    """
    Fixes some encoding issues in the dataset.
    Parameters:
      df:pd.DataFrame
    Returns:
      pd.DataFrame
    """
    for col in df.columns:
      df[col] = df[col].apply(lambda s: re.sub(r"(\xa00)|(_x000D_\n)|(\xa0)", " ",s))
    return df

  def df_to_json(self, df:pd.DataFrame)->list:
    """
    Turns the coicop df to a list containing dictionaries.
    Parameters:
      df:pd.DataFrame
    Returns:
      list
    """
    coicop_dict = df.to_dict(orient="index")
    return [coicop_dict[k] for k in coicop_dict.keys()]

  def load_dataset(self)->list:
    """
    Loads and preprocesses the coicop dataset. If no path is specified the data is downloaded from the official UN-Website. If a local path is set the dataset is loaded from that path.
    Returns:
      list

    """
    if self.path is not None:
      coicop_df = self.load_xlsx_dataset(self.path)
    else:
      try:
        coicop_df = self.load_xlsx_dataset(self.url)
      except Exception as e:
        print(e)
    coicop_df = self.fix_encoding(coicop_df)
    coicop_df["details"] = coicop_df.apply(lambda row: {"includes":row["includes"],"alsoIncludes": row["alsoIncludes"], "excludes":row["excludes"]},axis=1)
    coicop_df = coicop_df.drop(["includes", "alsoIncludes", "excludes"], axis=1)
    coicop_dict = self.df_to_json(coicop_df)
    return coicop_dict


