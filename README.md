# research_paper_idea_generation
This system is a Retrieval-Augmented Generation (RAG) application that generates novel, structured research ideas grounded in real academic papers.

Instead of relying purely on an LLM, the system first retrieves relevant research papers from a preprocessed dataset (like arXiv metadata), and then uses that context to guide the LLM in producing high-quality, realistic, and technically sound ideas.

 Workflow:

User enters a research topic

Topic is converted into an embedding

System retrieves top-k relevant papers using vector search

Retrieved abstracts are added as context to the prompt

LLM generates structured research ideas in JSON format

 Result: Ideas are context-aware, less hallucinated, and research-backed

 Technologies Used
🔹 1. Frontend

Streamlit

For building an interactive UI

🔹 2. Embedding Model

SentenceTransformers (all-MiniLM-L6-v2)

Converts text (abstracts + queries) into vector embeddings

🔹 3. Vector Database

FAISS

Stores embeddings and performs fast similarity search

🔹 4. LLM Inference

Groq API

Model: llama-3.3-70b-versatile

Generates structured research ideas based on retrieved context

🔹 5. Backend / Processing

Python

Core logic for data processing, retrieval, and orchestration

🔹 6. Data Source

arXiv-style JSON dataset

Contains:

Title

Abstract

Categories
