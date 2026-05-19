import random
import pandas as pd
from tqdm import tqdm

#corpus = {
#  "label":["examples", "examples", "list of all examples"]
#  "label2":["examples", "examples", "list of all examples"]
#}

def df_to_corpus(df:pd.DataFrame,text_column:str, label_column:str)->dict:
  
  unique_labels = df[label_column].unique()
  corpus = {}

  for u in tqdm(unique_labels):
    corpus.update(
      {
        u:df.query(f"{label_column} == '{u}'")[text_column].tolist()
      }
    )
  return unique_labels, corpus


def balanced_generator(batchsize: int, corpus: dict, unique_labels: list[str]):
    
    n_labels = len(unique_labels)
    k = min(batchsize, n_labels)

    while True:

      unique_label_ids = random.sample(range(n_labels), k)
      for label_id in unique_label_ids:
        label = unique_labels[label_id]
        examples = corpus[label]
        len_corpus = len(examples)
        if len_corpus == 1:
          yield {
              "anchor": examples[0],
              "positive": examples[0]
          }
        else:  
          if len_corpus > 2:
            idx_1, idx_2 = random.sample(range(len_corpus), 2)
            # When we have more than two examples in the class and when there are more than one unique entries, we need to ensure, that we don't have the same examples as positive AND anchor
            if len(list(set(examples))) > 1:
              while examples[idx_1] == examples[idx_2]:
                idx_2 = random.sample(range(len_corpus), 1)[0]
          else:
            idx_1, idx_2 = 0, 1
          yield {
              "anchor": examples[idx_1],
              "positive": examples[idx_2]
          }
