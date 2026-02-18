from extensions import db
from flask_login import UserMixin
from datetime import datetime

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)

class Bot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), default='stopped') # stopped, running, training
    config = db.Column(db.Text, nullable=True) # JSON string for config
    pnl = db.Column(db.Float, default=0.0)
    runtime = db.Column(db.String(50), default='0h 0m')
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status,
            'config': self.config,
            'pnl': self.pnl,
            'runtime': self.runtime,
            'last_updated': self.last_updated.isoformat()
        }
