import os
from flask import Flask, render_template, jsonify, request
from word_utils import WordVectorProcessor
import numpy as np
from models import db, WordCombination

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or "semantic-middle-word-key"

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all()

# Initialize word vector processor
word_processor = WordVectorProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/find_middle', methods=['POST'])
def find_middle():
    data = request.get_json()
    word1 = data.get('word1', '').lower().strip()
    word2 = data.get('word2', '').lower().strip()
    
    if not word1 or not word2:
        return jsonify({'error': 'Please provide both words'}), 400
    
    try:
        middle_word = word_processor.find_middle_word(word1, word2)
        # Get word vectors and similar words for visualization
        viz_data = word_processor.get_visualization_data(word1, word2, middle_word)
        # Get relationship explanation
        explanation = word_processor.explain_relationship(word1, word2, middle_word)
        
        # Save to history
        combination = WordCombination(
            word1=word1,
            word2=word2,
            middle_word=middle_word
        )
        db.session.add(combination)
        db.session.commit()
        
        return jsonify({
            'middle_word': middle_word,
            'visualization_data': viz_data,
            'explanation': explanation
        })
    except KeyError as e:
        return jsonify({'error': f'Word not found in vocabulary: {str(e)}'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_history():
    combinations = WordCombination.query.order_by(WordCombination.created_at.desc()).limit(10).all()
    return jsonify([combo.to_dict() for combo in combinations])
