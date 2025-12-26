# Policy Question Answering Assistant (RAG Mini Project)

## Setup Instructions

### 1. Clone Repository

git clone https://github.com/prathmeshpotdar/Rag

cd rag-policy-assistant

2. Create Virtual Environment

Copy code
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3. Install Dependencies

Copy code
pip install -r requirements.txt

4. Add API Key
Create a .env file:

txt
Copy code
OPENAI_API_KEY=your_api_key_here

5. Run
bash
Copy code
python rag.py

Architecture Overview
mathematica
Copy code
Policy Documents
      ↓
Text Loading & Chunking
      ↓
Embedding Generation
      ↓
Vector Store (FAISS)
      ↓
Top-K Semantic Retrieval
      ↓
Prompt-Constrained LLM
      ↓
Final Answer

Documents are chunked into overlapping segments

Chunks are embedded and stored in a FAISS vector index

Top-K relevant chunks are retrieved per question

The LLM is prompted to answer strictly from retrieved context

Prompts Used

Prompt Version 1 (Initial)
text
Copy code
You are a helpful assistant answering questions using company policy documents.
Use only the provided context to answer.
If the answer is not present, say so.

Context:
{context}

Question:
{question}

Answer:
Limitation: Weak hallucination control and no structured grounding.

Prompt Version 2 (Improved – Final)
text
Copy code
You are a policy question-answering assistant.

INSTRUCTIONS:
- Answer strictly using the provided context.
- Do NOT use outside knowledge.
- If the answer is missing or unclear, explicitly say so.

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
Improvement: Stronger hallucination control, clearer structure, and enforced grounding.

Evaluation Results
Evaluation Questions
Question	Result
What is the refund window?	✅
Are shipping fees refundable?	⚠️
How long does express shipping take?	✅
Do you support international returns?	✅ (Correctly refused)
Is cash on delivery available?	✅ (Out of scope handled)

Scoring Legend
✅ Correct & grounded

⚠️ Partially answerable

❌ Hallucinated / incorrect

Summary:
The system answered grounded questions accurately and correctly refused to answer questions outside the document scope.

Key Trade-offs & Improvements
Trade-offs Made
Used simple chunking instead of structure-aware chunking

Relied on semantic retrieval without reranking

Manual evaluation instead of automated scoring

Improvements with More Time
Add reranking using a cross-encoder

Introduce automated evaluation (LLM-as-judge)

Add citation IDs per retrieved chunk

Implement prompt versioning and tracing

Improve chunking using document headings
