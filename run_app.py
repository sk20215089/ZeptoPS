import streamlit as st
import faiss
import numpy as np
import pandas as pd
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

# Load the FAISS index
index = faiss.read_index('product_embeddings.index')

# Load the metadata
df_subset = pd.read_pickle('product_metadata.pkl')

# function to get BERT embeddings
def get_bert_embeddings(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=512)
    outputs = model(inputs)
    pooled_output = outputs.last_hidden_state[:, 0, :].numpy()

    return pooled_output.astype('float32')

# Define semantic search with FAISS
def semantic_search_with_faiss(query, top_n=5):
    query_embedding = get_bert_embeddings(query)
    distances, top_indices = index.search(query_embedding, top_n)
    return df_subset.iloc[top_indices[0]][['product_name', 'description', 'product_category_tree']]

# Streamlit UI
st.title("Zepto Query Search")
st.subheader("Search for similar products")
query = st.text_input("Enter your search query here:")

#trigger the search
if st.button("Submit"):
    if query:
        results = semantic_search_with_faiss(query)
        st.markdown("**Search Results:**")
        st.dataframe(results)

