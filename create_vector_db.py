"""
Vector Database Creator for Portfolio Files
Converts PDFs and text files into a searchable vector database
"""

import os
from pathlib import Path
import pickle
from typing import List

# Install these if needed: pip install sentence-transformers PyPDF2 faiss-cpu --break-system-packages
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

try:
    import PyPDF2
except ImportError:
    print("Installing PyPDF2...")
    os.system("pip install PyPDF2 --break-system-packages")
    import PyPDF2


class DocumentProcessor:
    def __init__(self, folder_path: str = "my_files", chunk_size: int = 400, overlap: int = 100):
        """
        Initialize document processor
        
        Args:
            folder_path: Path to documents folder
            chunk_size: Number of words per chunk (default: 400)
            overlap: Number of overlapping words between chunks (default: 100)
        """
        self.folder_path = folder_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.documents = []
        self.embeddings = None
        self.index = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"📝 Chunk settings: {chunk_size} words per chunk, {overlap} word overlap")
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
        return text
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from a text file"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading {txt_path}: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks using configured size"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def load_documents(self):
        """Load all documents from the folder"""
        print(f"Loading documents from {self.folder_path}...")
        
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Folder '{self.folder_path}' not found!")
        
        for filename in os.listdir(self.folder_path):
            filepath = os.path.join(self.folder_path, filename)
            
            if filename.lower().endswith('.pdf'):
                print(f"Processing PDF: {filename}")
                text = self.extract_text_from_pdf(filepath)
                chunks = self.chunk_text(text)
                
                for i, chunk in enumerate(chunks):
                    self.documents.append({
                        'text': chunk,
                        'source': filename,
                        'chunk_id': i,
                        'type': 'pdf'
                    })
            
            elif filename.lower().endswith('.txt'):
                print(f"Processing text file: {filename}")
                text = self.extract_text_from_txt(filepath)
                chunks = self.chunk_text(text)
                
                for i, chunk in enumerate(chunks):
                    self.documents.append({
                        'text': chunk,
                        'source': filename,
                        'chunk_id': i,
                        'type': 'txt'
                    })
        
        print(f"Loaded {len(self.documents)} chunks from {len(set(d['source'] for d in self.documents))} files")
    
    def create_embeddings(self):
        """Create embeddings for all documents"""
        print("Creating embeddings...")
        texts = [doc['text'] for doc in self.documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Created {len(self.embeddings)} embeddings")
    
    def build_faiss_index(self):
        """Build FAISS index from embeddings"""
        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def save_database(self, output_dir: str = "vector_db"):
        """Save the vector database and documents"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(output_dir, "faiss_index.bin"))
        
        # Save documents metadata
        with open(os.path.join(output_dir, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save embeddings
        np.save(os.path.join(output_dir, "embeddings.npy"), self.embeddings)
        
        print(f"Vector database saved to '{output_dir}/'")
        print(f"  - faiss_index.bin: FAISS index")
        print(f"  - documents.pkl: Document metadata")
        print(f"  - embeddings.npy: Embedding vectors")
    
    def search(self, query: str, top_k: int = 3):
        """Search for similar documents"""
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append({
                'document': self.documents[idx],
                'distance': float(distance)
            })
        
        return results


def main():
    """
    Create vector database from documents
    
    Chunk size recommendations:
    - Small (300 words): More precise retrieval, more chunks
    - Medium (400 words): Balanced - RECOMMENDED  
    - Large (600 words): More context per chunk, fewer chunks
    
    Overlap recommendations:
    - 50-100 words: Good for maintaining context between chunks
    """
    
    print("="*60)
    print("📚 VECTOR DATABASE CREATOR")
    print("="*60)
    
    # Configuration - adjust these for optimization!
    CHUNK_SIZE = 1200      # Words per chunk (try 300-600)
    CHUNK_OVERLAP = 200  # Overlapping words (try 50-150)
    
    # Create processor with custom settings
    processor = DocumentProcessor(
        folder_path="my_files",
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )
    
    # Process documents
    processor.load_documents()
    processor.create_embeddings()
    processor.build_faiss_index()
    
    # Save database
    processor.save_database(output_dir="vector_db")
    
    print("\n✅ Vector database created successfully!")
    print(f"   📊 Total chunks: {len(processor.documents)}")
    print(f"   📝 Chunk size: {CHUNK_SIZE} words")
    print(f"   🔄 Overlap: {CHUNK_OVERLAP} words")
    print("\nYou can now use this database in your chatbot.")
    
    # Test search
    print("\n--- Testing Search ---")
    test_query = input("Enter a test query (or press Enter to skip): ").strip()
    if test_query:
        results = processor.search(test_query, top_k=5)
        print(f"\nTop 5 results for '{test_query}':")
        for i, result in enumerate(results, 1):
            similarity = 1 / (1 + result['distance'])
            print(f"\n{i}. Source: {result['document']['source']}")
            print(f"   Similarity: {similarity:.2f}")
            print(f"   Text preview: {result['document']['text'][:150]}...")


if __name__ == "__main__":
    main()
