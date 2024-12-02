from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize the sentence transformer model
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Store QA pairs mapping
        self.qa_mapping = []
        
    def add_qa_pairs(self, qa_pairs: List[Dict]):
        """Add QA pairs to the vector store"""
        if not qa_pairs:
            return
            
        # Create embeddings for questions
        questions = [qa['question'] for qa in qa_pairs]
        embeddings = self.model.encode(questions, show_progress_bar=True)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Update QA mapping
        start_idx = len(self.qa_mapping)
        for idx, qa in enumerate(qa_pairs):
            self.qa_mapping.append({
                'index': start_idx + idx,
                'qa_pair': qa
            })
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar questions"""
        # Create query embedding
        query_embedding = self.model.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            k
        )
        
        # Get results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.qa_mapping):
                result = self.qa_mapping[idx]['qa_pair'].copy()
                result['similarity_score'] = float(1 / (1 + distance))  # Convert distance to similarity score
                results.append(result)
        
        return results
    
    def save(self, space_key: str):
        """Save vector store to disk"""
        save_path = self.cache_dir / f"vector_store_{space_key}"
        save_path.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "faiss_index.bin"))
        
        # Save QA mapping
        with open(save_path / "qa_mapping.pkl", 'wb') as f:
            pickle.dump(self.qa_mapping, f)
            
        logger.info(f"Vector store saved to {save_path}")
    
    def load(self, space_key: str) -> bool:
        """Load vector store from disk"""
        save_path = self.cache_dir / f"vector_store_{space_key}"
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(save_path / "faiss_index.bin"))
            
            # Load QA mapping
            with open(save_path / "qa_mapping.pkl", 'rb') as f:
                self.qa_mapping = pickle.load(f)
                
            logger.info(f"Vector store loaded from {save_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load vector store: {str(e)}")
            return False 