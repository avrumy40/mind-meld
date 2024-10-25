import numpy as np
from gensim.models import KeyedVectors
from gensim.downloader import load
from sklearn.decomposition import PCA
from typing import Dict, List, Union, Any

class WordVectorProcessor:
    def __init__(self):
        # Load a small pre-trained word2vec model
        print("Loading word vectors...")
        self.model: KeyedVectors = load('glove-wiki-gigaword-50')  # type: ignore
        print("Word vectors loaded!")
        # Initialize PCA for dimensionality reduction
        self.pca = PCA(n_components=2)

    def find_middle_word(self, word1: str, word2: str) -> str:
        # Verify words exist in vocabulary
        if word1 not in self.model:
            raise KeyError(word1)
        if word2 not in self.model:
            raise KeyError(word2)
        
        # Get word vectors
        vec1 = self.model.get_vector(word1)
        vec2 = self.model.get_vector(word2)
        
        # Calculate middle vector
        middle_vec = (vec1 + vec2) / 2
        
        # Find most similar word to middle vector
        result = self.model.similar_by_vector(middle_vec, topn=1)
        middle_word = result[0][0]
        
        # Avoid returning one of the input words
        if middle_word in [word1, word2]:
            result = self.model.similar_by_vector(middle_vec, topn=5)
            for word, _ in result:
                if word not in [word1, word2]:
                    middle_word = word
                    break
        
        return middle_word

    def explain_relationship(self, word1: str, word2: str, middle_word: str) -> Dict[str, Any]:
        # Calculate cosine similarities
        similarity_1_2 = self.model.similarity(word1, word2)
        similarity_1_m = self.model.similarity(word1, middle_word)
        similarity_2_m = self.model.similarity(word2, middle_word)
        
        # Get most similar contexts for each word
        context1 = [word for word, _ in self.model.similar_by_word(word1, topn=5)]
        context2 = [word for word, _ in self.model.similar_by_word(word2, topn=5)]
        context_m = [word for word, _ in self.model.similar_by_word(middle_word, topn=5)]
        
        # Create explanation
        explanation = {
            'similarities': {
                'between_inputs': round(float(similarity_1_2) * 100, 2),
                'word1_to_middle': round(float(similarity_1_m) * 100, 2),
                'word2_to_middle': round(float(similarity_2_m) * 100, 2)
            },
            'contexts': {
                'word1': context1,
                'word2': context2,
                'middle': context_m
            }
        }
        
        return explanation

    def get_visualization_data(self, word1: str, word2: str, middle_word: str) -> Dict[str, List[Dict[str, Any]]]:
        # Get similar words to create a richer visualization
        similar_words1 = self.model.similar_by_word(word1, topn=3)
        similar_words2 = self.model.similar_by_word(word2, topn=3)
        
        # Collect all words and their vectors
        words = [word1, word2, middle_word]
        words.extend([w for w, _ in similar_words1])
        words.extend([w for w, _ in similar_words2])
        words = list(set(words))  # Remove duplicates
        
        # Get vectors for all words
        vectors = [self.model.get_vector(w) for w in words]
        
        # Reduce dimensionality to 2D for visualization
        coords = self.pca.fit_transform(vectors)
        
        # Create nodes and edges data
        nodes = []
        edges = []
        
        for i, (word, coord) in enumerate(zip(words, coords)):
            nodes.append({
                'id': word,
                'x': float(coord[0]),
                'y': float(coord[1]),
                'type': 'input' if word in [word1, word2] else 
                        'middle' if word == middle_word else 'similar'
            })
        
        # Add edges between input words and middle word
        edges.extend([
            {'source': word1, 'target': middle_word},
            {'source': word2, 'target': middle_word}
        ])
        
        # Add edges between input words and their similar words
        for w, _ in similar_words1:
            edges.append({'source': word1, 'target': w})
        for w, _ in similar_words2:
            edges.append({'source': word2, 'target': w})
        
        return {
            'nodes': nodes,
            'edges': edges
        }
