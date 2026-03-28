import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

#Load faiss index
index = faiss.read_index("index.faiss")
with open("bm25_corpus.pkl", "rb") as f:
    tokenized_corpus= pickle.load(f)

#Load id_to_doc mapping
with open("id_to_doc.json", "r") as f:
    id_to_doc = json.load(f)

#Load embeddings model
model = SentenceTransformer("all-MiniLM-L6-v2")

bm25 = BM25Okapi(tokenized_corpus)
#Build retrieval function

def hybrid_retrieve(query, k=20):
    #embedding search
    query_embedding = model.encode([query])

    distances, indices = index.search(np.array(query_embedding),k)
    # print(f"distances: {distances} indices: {indices}")

    embeddings_result = []
    for rank,idx in enumerate(indices[0]):
        embeddings_result.append({
            "doc": idx,
            "score": 1/(1+distances[0][rank])
        })

    #bm 25 search
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    # print(f"bm25_scores: {bm25_scores}")

    bm25_results = []
    for i, score in enumerate(bm25_scores):
        bm25_results.append({
            "doc": i,
            "score": score
        })
    # print(f"bm25_results: {bm25_results}")

    #normalize scores
    def normalize_scores(scores):
        vals = [s["score"] for s in scores]
        min_v, max_v = min(vals), max(vals)
        for s in scores:
            s["score"] = (s["score"] - min_v) / (max_v - min_v + 1e-8)
        return scores

    embeddings_result = normalize_scores(embeddings_result)
    bm25_results = normalize_scores(bm25_results)

    #combine results
    combined = {}

    for item in embeddings_result:
        combined[item["doc"]] = combined.get(item["doc"], 0) + 0.7 * item["score"]

    for item in bm25_results:
        combined[item["doc"]] = combined.get(item["doc"], 0) + 0.3 * item["score"]

    sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    top_docs = [id_to_doc[str(idx)] for idx, _ in sorted_docs[:k]]

    return top_docs


#format context for LLM
def format_context(docs):
    context = ""
    for d in docs:
        context += d["text"] + "\n\n"
    return context


def rerank(query, docs, top_k=5):
    pairs = [(query, doc["text"]) for doc in docs]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))

    #sort by cross-encoder scores
    sorted_docs = sorted(scored_docs, key=lambda x:x[1], reverse=True)

    return [doc for doc, _ in sorted_docs[:top_k]]


def retrieve(query):

    #step 1: hybrid search using embeddings search + bm25 algorithm
    candidates = hybrid_retrieve(query, k=20)

    #step 2: rerank using cross-encoder for precise results
    final_docs = rerank(query, candidates, top_k=5)

    return final_docs
