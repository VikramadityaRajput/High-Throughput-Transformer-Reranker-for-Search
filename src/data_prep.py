import pandas as pd
import os 

def load_data(data_dir: str = "../data/raw") -> pd.DataFrame:
#ESCI means Exact, Substitute, Complement, Irrelevant.
#We are trying to link user searches to product catalog items
    ex_url = "https://github.com/amazon-science/esci-data/blob/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
    product_url = "https://github.com/amazon-science/esci-data/blob/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet"
    #download and read the files and then put them into a dataframe
    examples_df = pd.read_parquet(ex_url)
    product_df = pd.read_parquet(product_url)
    #filter out to only get US data
    examples_df = examples_df[examples_df["product_locale"] == "us"]
    merged_df = pd.merge(examples_df, product_df[['product_id', 'product_title', 'product_description']], on='product_id', how='left') 
    mapping = {'E': 3, 'S': 2, 'C' : 1, 'I' : 0}
    merged_df['relevance_score'] = merged_df['esci_label'].map(mapping) #create a new column with the relevant value

    os.makedirs(data_dir, exist_ok=True) 
    sample = merged_df.sampled(n=100000, random_state=67) #100000 is a good amount and 67 is tuff
    sample.to_parquet(f"{data_dir}/esci_sample.parquet") #save the data
    print("Saved the Sample")
    return sample

if __name__ == "__main__":
    df = load_data()
    print(df[['query', 'product_title', 'esci_label', 'relevance_score']].head())