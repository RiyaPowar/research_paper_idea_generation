import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

with open('/Users/wynum_air_14/xyz/research_paper_idea_generation/data/top_10000_records.json','r') as f:
    data = json.load(f)

#Text per document
def build_text(doc):
    return f"""
    Title: {doc.get('title','')}
    Category: {doc.get('categories','')}
    Abstarct: {doc.get('abstract','')}
"""

documents = []
for doc in data:
    documents.append({
        "id":doc["id"],
        "text": build_text(doc)
    })

#Generate embeddings
model= SentenceTransformer("all-MiniLM-L6-v2")

texts = [doc["text"] for doc in documents]
# print(f"texts:::::::::{texts}")

embeddings = model.encode(texts)
embeddings = np.array(embeddings)

print(f"embeddings:::::::::{embeddings}")


#Store in FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)
# print("indexes:::::::::",index)

tokenized_corpus = [doc["text"].lower().split() for doc in documents]

with open("bm25_corpus.pkl", "wb") as f:
    pickle.dump(tokenized_corpus, f)

#ID mapping
id_to_doc = {i: documents[i] for i in range(len(documents))}

np.save("embeddings.npy", embeddings)
faiss.write_index(index, "index.faiss")

with open("id_to_doc.json", "w") as f:
    json.dump(id_to_doc, f)

print("Data processing and embedding generation completed.")
