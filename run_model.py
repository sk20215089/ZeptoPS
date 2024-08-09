import faiss
import numpy as np
import pandas as pd
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import streamlit as st

# Load the FAISS index
index = faiss.read_index('product_embeddings.index')

# Load the metadata
df_subset = pd.read_pickle('product_metadata.pkl')

#  function to get BERT embeddings
def get_bert_embeddings(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=512)
    outputs = model(inputs)
    pooled_output = outputs.last_hidden_state[:, 0, :].numpy()
    return pooled_output.astype('float32')

#function to do semantic search
def semantic_search_with_faiss(query, top_n=5):
    query_embedding = get_bert_embeddings(query)  
    distances, top_indices = index.search(query_embedding, top_n)
    results = df_subset.iloc[top_indices[0]].copy()
    results['similarity'] = distances[0]
    results = results.sort_values(by='similarity', ascending=True)
    
    return results[['product_name', 'description', 'product_category_tree']]
# Streamlit app
st.title("Zepto Query Search")
st.write("Search for similar products")

query = st.text_input("Enter your search query here:", "")
submit_button = st.button("Submit")

if submit_button:
    if query:
        results_with_faiss = semantic_search_with_faiss(query)
        st.write("Search Results:")
        st.dataframe(results_with_faiss)  
    else:
        st.write("Please enter a query to search.")
