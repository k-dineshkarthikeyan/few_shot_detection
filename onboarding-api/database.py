from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy import create_engine, Column, Integer, Numeric, ARRAY
import os

url=r'postgresql://postgres:dinesh@localhost/one'
engine = create_engine(url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Embedding(Base):
    __tablename__ = 'embeddings'
    id = Column(Integer, primary_key=True)
    encodings = Column(ARRAY(Numeric))

def read_all_embeddings(db: Session):
    result = db.query(Embedding).all()
    return result

def read_one_embedding(db: Session, id:int):
    result = db.query(Embedding).filter(Embedding.id == id).first()
    return result

def put_embeddings(db: Session, encodings: list[list[int]]):
    item = Embedding(encodings = encodings)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item

def update_embeddings(db: Session, id: int, encodings: list[list[int]]):
    item = db.query(Embedding).filter(Embedding.id == id).first()
    item.encodings = Embedding(id = id, encodings = encodings).encodings
    db.commit()
    db.refresh(item)
    return item
