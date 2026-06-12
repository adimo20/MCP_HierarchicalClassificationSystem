import pandas as pd 

# File used has been downloaded from here: https://showvoc.op.europa.eu/#/datasets/ESTAT_Statistical_Classification_of_Economic_Activities_in_the_European_Community_Rev._2.1._%28NACE_2.1%29/downloads
# Filename: NACE_Rev2.1_Structure_Explanatory_Notes_EN.xlsx

def load_nace(
    path:str
)->list[dict]:
    """Loads the official NACE documentation from an xlsx file and transforms it into a structure we can load with Code/ClassificationSystem"""
    
    df = pd.read_excel(path, dtype=str)
    colnames = ["ID", "HEADING", "PARENT_ID", "Includes", "LEVEL", "IncludesAlso", "Excludes"]
    
    df = df[colnames]
    df.columns = ["code", "description", "parent_code", "Includes", "level", "IncludesAlso", "Excludes"]

    for c in df.columns:
        df[c] = df[c].fillna('')
    
    df["details"] = df.apply(lambda row: {"parent_code" : row["parent_code"], "includes" :  row["Includes"], "alsoIcludes" : row["IncludesAlso"], "excludes" : row["Excludes"]}, axis=1)
    df = df[["code", "description", "level", "details"]]


    nace_dict = df.to_dict(orient="index")
    nace_list = [nace_dict[k] for k in nace_dict.keys()]
    return nace_list
