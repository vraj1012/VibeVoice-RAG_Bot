from sentence_transformers import SentenceTransformer
import chromadb

class MemoryEngine:
    def __init__(self):
        print("⏳ Loading RAG Memory...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path="my_rag_db")
        self.collection = self.client.get_or_create_collection(name="avocado_knowledge")
        self.history = []

    def get_context(self, text):
        query_embedding = self.embedder.encode(text).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=1)
        if results['documents'][0]:
            return "\n".join(results['documents'][0])
        return ""

    def get_recent_history(self):
        return "\n".join(self.history[-6:])
    
    def add_to_history(self, entry):
        self.history.append(entry)