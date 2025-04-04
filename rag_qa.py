from transformers import pipeline
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd 
import os
import time

hf_token = os.getenv("Used token from hugging face")

df = pd.read_csv("cleaned_hotel_bookings.csv")


# Load FAISS index
index = faiss.read_index("hotel_bookings.index")

# Load Sentence Transformer model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load a small open-source LLM (Llama2, Falcon, etc.)
qa_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1",token=hf_token)

# Function to retrieve most relevant booking records
def retrieve_relevant_bookings(query, k=3):
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)
    
    # Search in FAISS index
    _, indices = index.search(query_embedding, k)
    
    # Fetch matching bookings
    results = df.iloc[indices[0]].to_dict(orient="records")
    return results

# Generate answer using LLM
def generate_answer(query):
    relevant_data = retrieve_relevant_bookings(query)
    
    # Convert retrieved data to text format
    context = "\n".join([str(item) for item in relevant_data])
    
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    response = qa_pipeline(prompt, max_new_tokens=100, do_sample=True)
    return response[0]["generated_text"]


