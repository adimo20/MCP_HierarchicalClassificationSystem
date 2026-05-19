from sentence_transformers.evaluation import InformationRetrievalEvaluator

def setup_evaluator(val_df, text_column, label_column):
    queries = {}
    corpus = {}
    relevant_docs = {}
    
    query_size = min(35000, len(val_df) // 2)
    query_df = val_df.sample(n=query_size, random_state=42)
    corpus_df = val_df.drop(query_df.index)

    print(f"Setup: {len(query_df)} queries, {len(corpus_df)} corpus docs")

    class_to_ids = corpus_df.groupby(text_column).groups
    class_to_ids = {k: [str(i) for i in v] for k, v in class_to_ids.items()}
    
    for i, row in query_df.iterrows():
        query_id = f"q_{i}"
        category = row[text_column]
        
        pos_ids = class_to_ids.get(category, [])
        
        if len(pos_ids) > 0:
            queries[query_id] = row[label_column]
            relevant_docs[query_id] = set(pos_ids)
    
    corpus = {str(i): row[label_column] for i, row in corpus_df.iterrows()}
    
    print(f"Final: {len(queries)} queries with relevant docs, {len(corpus)} corpus docs")
    
    evaluator = InformationRetrievalEvaluator(
        queries, corpus, relevant_docs, 
        name='SEA5-retrieval-eval',
        mrr_at_k=[10], 
        accuracy_at_k=[1, 5, 10, 20]
    )
    return evaluator

