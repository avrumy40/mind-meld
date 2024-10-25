from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class WordCombination(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    word1 = db.Column(db.String(100), nullable=False)
    word2 = db.Column(db.String(100), nullable=False)
    middle_word = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'word1': self.word1,
            'word2': self.word2,
            'middle_word': self.middle_word,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }
