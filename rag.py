from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# -------------------------
# Load embeddings + FAISS
# -------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "faiss_index",
    embedding,
    allow_dangerous_deserialization=True
)

# -------------------------
# LLM (FREE - HuggingFace)
# -------------------------
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256,
    temperature=0.3
)

llm = HuggingFacePipeline(pipeline=pipe)

# -------------------------
# Retrieval Function (Improved)
# -------------------------
def retrieve_docs(query):
    docs_with_scores = db.similarity_search_with_score(query, k=3)

    filtered_docs = []
    for doc, score in docs_with_scores:
        if score < 1.5:
            filtered_docs.append(doc)

    # fallback if empty
    if len(filtered_docs) == 0:
        filtered_docs = [doc for doc, _ in docs_with_scores[:2]]

    return filtered_docs[:2]

# -------------------------
# Main Loop
# -------------------------
while True:
    query = input("\nAsk: ")

    docs = retrieve_docs(query)

    # Debug: show retrieved chunks
    print("\n--- Retrieved Context ---")
    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1}:\n{doc.page_content[:200]}")

    # Build context (IMPORTANT)
    context = "\n\n".join([doc.page_content for doc in docs])
    context = context[:1000]   # prevent token overflow

    # Prompt
    prompt = f"""
You are an AI tutor.

Instructions:
- Answer ONLY using the context
- If answer is not in context, say: Not in document
- Be clear and structured
- Do not add extra knowledge

Context:
{context}

Question:
{query}

Answer in this format:
- Explanation:
- Key Points:
"""

    # Call LLM
    response = llm.invoke(prompt)

    print("\nAnswer:\n", response)