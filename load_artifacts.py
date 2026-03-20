import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

#Load faiss index
index = faiss.read_index("index.faiss")

#Load id_to_doc mapping
with open("id_to_doc.json", "r") as f:
    id_to_doc = json.load(f)

#Load embeddings model
model = SentenceTransformer("all-MiniLM-L6-v2")


#Build retrieval function

def retrieve(query, k=5):
    query_embedding = model.encode([query])

    distances, indices = index.search(np.array(query_embedding),k)

    results = [id_to_doc[str(i)] for i in indices[0]]
    # print(f"results:::::::{results}")
    return results



#format context for LLM
def format_context(docs):
    context = ""
    for d in docs:
        context += d["text"] + "\n\n"
    return context
