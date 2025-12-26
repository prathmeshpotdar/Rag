import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# ===============================
# CONFIG
# ===============================
DATA_DIR = "data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 3

PROMPT_V2 = """
You are a policy question-answering assistant.

INSTRUCTIONS:
- Answer strictly using the provided context.
- Do NOT use outside knowledge.
- If the answer is missing or unclear, explicitly say so.
- Keep the answer concise and factual.

FORMAT:
Answer:
- <your answer>

Supporting Policy Excerpt:
- "<quoted text>"

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

# ===============================
# SETUP
# ===============================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ===============================
# DOCUMENT LOADING
# ===============================
def load_documents(folder):
    texts = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if file.endswith(".txt") or file.endswith(".md"):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())

        elif file.endswith(".pdf"):
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            texts.append(text)

    return texts

# ===============================
# CHUNKING
# ===============================
def chunk_text(text, chunk_size=500, overlap=80):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks

# ===============================
# VECTOR STORE
# ===============================
def build_vector_store(chunks):
    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index, embeddings

def retrieve(query, index, chunks, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

# ===============================
# LLM CALL
# ===============================
def generate_answer(question, context):
    prompt = PROMPT_V2.format(
        context=context,
        question=question
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

# ===============================
# MAIN QA PIPELINE
# ===============================
def answer_question(question, index, chunks):
    retrieved_chunks = retrieve(question, index, chunks)

    if not retrieved_chunks:
        return "No relevant policy information found."

    context = "\n\n".join(retrieved_chunks)
    return generate_answer(question, context)

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    print("Loading documents...")
    documents = load_documents(DATA_DIR)

    print("Chunking documents...")
    chunks = []
    for doc in documents:
        chunks.extend(chunk_text(doc))

    print(f"Total chunks: {len(chunks)}")

    print("Building vector store...")
    index, _ = build_vector_store(chunks)

    print("\nRAG System Ready. Ask questions (type 'exit' to quit).\n")

    while True:
        q = input("Question: ")
        if q.lower() == "exit":
            break

        answer = answer_question(q, index, chunks)
        print("\nAnswer:\n", answer)
        print("-" * 60)
