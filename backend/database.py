from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import pickle
import numpy as np

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True, index=True)
    name = Column(String)
    embedding = Column(LargeBinary)

class Attendance(Base):
    __tablename__ = 'attendance'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

engine = create_engine('sqlite:///backend/erp_face.db', connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def save_embedding(session, user_id: str, name: str, emb: np.ndarray):
    """Insert or update embedding for a user."""
    init_db()
    data = pickle.dumps(emb.astype('float32'))
    user = session.query(User).filter(User.user_id == user_id).first()
    if user:
        user.embedding = data
        user.name = name
    else:
        user = User(user_id=user_id, name=name, embedding=data)
        session.add(user)
    session.commit()

def get_all_embeddings(session):
    init_db()
    users = session.query(User).all()
    items = []
    for u in users:
        emb = pickle.loads(u.embedding)
        items.append((u.user_id, u.name, emb))
    return items
