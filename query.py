from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

query = input("Ask something: ")

docs = db.similarity_search(query, k=3)

for i, doc in enumerate(docs):
    print(f"\nResult {i+1}:")
    print(doc.page_content)