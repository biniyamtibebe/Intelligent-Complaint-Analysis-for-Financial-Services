# notebooks/task2_chunking_embedding_indexing.py

import pandas as pd
import os
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# -----------------------------
# 1. Load the cleaned dataset
# -----------------------------
data_path = r"C:\Users\hp\Pictures\financial intelligence\Intelligent-Complaint-Analysis-for-Financial-Services\rag-complaint-chatbot\data\processed\filtered_complaints.csv"
df = pd.read_csv(data_path)

print(f"Loaded {len(df)} complaints after filtering and cleaning.")

# Ensure product column exists and map to consistent categories for CrediTrust
# Adjust these mappings based on actual values in your filtered data
product_mapping = {
    'Credit card': 'Credit Cards',
    'Credit card or prepaid card': 'Credit Cards',
    'Consumer Loan': 'Personal Loans',
    'Vehicle loan or lease': 'Personal Loans',  # sometimes included
    'Bank account or service': 'Savings Accounts',
    'Checking or savings account': 'Savings Accounts',
    'Money transfer': 'Money Transfers',
    'Money transfers': 'Money Transfers',
    'Virtual currency': 'Money Transfers',  # optional inclusion
}

# Create a standardized product_category column
df['product_category'] = df['Product'].map(product_mapping)

# Drop rows where product_category is NaN (not in our target products)
df = df.dropna(subset=['product_category']).reset_index(drop=True)

print("Product distribution after mapping:")
print(df['product_category'].value_counts())

# -----------------------------
# 2. Create Stratified Sample (10,000 - 15,000 complaints)
# -----------------------------
SAMPLE_SIZE = 12000  # Target size — adjust if needed

# Stratified sampling proportional to product_category
stratified_sample = df.groupby('product_category', group_keys=False).apply(
    lambda x: x.sample(frac=SAMPLE_SIZE / len(df), random_state=42)
)

# If frac results in more/less than desired, cap or upsample
if len(stratified_sample) > SAMPLE_SIZE:
    stratified_sample = stratified_sample.sample(n=SAMPLE_SIZE, random_state=42)
elif len(stratified_sample) < SAMPLE_SIZE:
    # Upsample minority classes if needed
    stratified_sample = stratified_sample.sample(n=SAMPLE_SIZE, replace=True, random_state=42)

stratified_sample = stratified_sample.reset_index(drop=True)

print(f"\nStratified sample created: {len(stratified_sample)} complaints")
print("Sample distribution:")
print(stratified_sample['product_category'].value_counts())

# -----------------------------
# 3. Text Chunking Strategy
# -----------------------------
# We use RecursiveCharacterTextSplitter — it tries to split on paragraphs, then sentences, etc.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,          # ~100-150 words per chunk (good for MiniLM context)
    chunk_overlap=50,        # Small overlap helps maintain context across chunks
    length_function=len,     # Measure length in characters
    separators=["\n\n", "\n", " ", ""]  # Default recursive order
)

# Test on one example
example_text = stratified_sample['cleaned_narrative'].iloc[0]
chunks_example = text_splitter.split_text(example_text)
print(f"\nExample chunks (first complaint): {len(chunks_example)} chunks created")

# -----------------------------
# 4. Load Embedding Model
# -----------------------------
# all-MiniLM-L6-v2: 384-dim, fast, excellent performance on semantic similarity
# Trained on 1B+ sentence pairs — ideal for complaint retrieval
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Chroma will use the same model via huggingface embedding function
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# -----------------------------
# 5. Prepare Chunks with Metadata
# -----------------------------
chunks = []
metadatas = []
ids = []
documents = []  # raw text chunks

print("\nChunking all narratives...")
for idx, row in tqdm(stratified_sample.iterrows(), total=len(stratified_sample)):
    narrative = row['cleaned_narrative']
    complaint_id = str(row.get('Complaint ID', f"complaint_{idx}"))  # Use actual ID if available
    
    # Split into chunks
    text_chunks = text_splitter.split_text(narrative)
    
    for chunk_idx, chunk_text in enumerate(text_chunks):
        if len(chunk_text.strip()) < 20:  # Skip very tiny chunks
            continue
            
        chunk_id = f"{complaint_id}_chunk{chunk_idx}"
        
        ids.append(chunk_id)
        documents.append(chunk_text)
        metadatas.append({
            "complaint_id": complaint_id,
            "product_category": row['product_category'],
            "product": row['Product'],
            "date_received": str(row.get('Date received', '')),
            "chunk_index": chunk_idx,
            "total_chunks": len(text_chunks),
            "source_narrative_length": len(narrative)
        })

print(f"Total chunks created: {len(documents)}")

# -----------------------------
# 6. Create and Populate ChromaDB Vector Store
# -----------------------------
persist_directory = "vector_store/chroma_sample_index"

# Initialize Chroma client with persistence
client = chromadb.PersistentClient(path=persist_directory)

# Create or get collection
collection = client.get_or_create_collection(
    name="complaint_chunks_sample",
    embedding_function=embedding_function,  # Chroma will embed on add if needed
    metadata={"hnsw:space": "cosine"}      # Cosine similarity — best for text embeddings
)

# Add in batches to avoid memory issues
batch_size = 500
print("\nAdding chunks to ChromaDB in batches...")
for i in tqdm(range(0, len(documents), batch_size)):
    batch_docs = documents[i:i+batch_size]
    batch_ids = ids[i:i+batch_size]
    batch_metas = metadatas[i:i+batch_size]
    
    collection.add(
        documents=batch_docs,
        ids=batch_ids,
        metadatas=batch_metas
    )

print(f"Vector store successfully created and persisted at: {persist_directory}")
print(f"Collection contains {collection.count()} vectors.")

# -----------------------------
# 7. Quick Sanity Check: Perform a Test Query
# -----------------------------
test_results = collection.query(
    query_texts=["late fees charged unfairly on credit card"],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

print("\n--- Test Retrieval Results ---")
for doc, meta, dist in zip(test_results['documents'][0], 
                          test_results['metadatas'][0], 
                          test_results['distances'][0]):
    print(f"Distance: {dist:.4f}")
    print(f"Product: {meta['product_category']}")
    print(f"Snippet: {doc[:200]}...\n")