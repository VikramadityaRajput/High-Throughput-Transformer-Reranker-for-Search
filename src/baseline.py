import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np

def bm25_search(data_path="../data/raw/esci_sample.parquet"):
    # Using the safe absolute path method
    df = pd.read_parquet(data_path)
    target_query = "wireless mouse"
    query_data = df[df['query'] == target_query].copy()
    
    if query_data.empty:
        print(f"Query '{target_query}' not found in the sample. Try another one!")
        return
        
    print(f"Found {len(query_data)} candidate products to sort.")
    tokenized_corpus = [str(title).lower().split(" ") for title in query_data['product_title']]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = target_query.lower().split(" ")
    scores = bm25.get_scores(tokenized_query)
    query_data['bm25_score'] = scores
    
    top_5 = query_data.sort_values(by='bm25_score', ascending=False).head(5)
    for index, row in top_5.iterrows():
        shortened_title = str(row['product_title'])[:80] + "..."
        print(f"Score: {row['bm25_score']:.2f} | Label: {row['esci_label']} | Title: {shortened_title}")

if __name__ == "__main__":
    bm25_search()