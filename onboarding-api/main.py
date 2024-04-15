from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
from utils import pythonise_embeddings, convert_to_postgres
from database import Base, SessionLocal, put_embeddings, engine, read_one_embedding, update_embeddings
from model import SiameseNetwork
from pydantic import BaseModel

Base.metadata.create_all(bind=engine)

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SiameseNetwork()
pretrained = torch.load('model1.pt', map_location=device)
# pretrained = torch.load('./final_training_model.pt', map_location=device)
model.load_state_dict(pretrained)
model.eval()

class UpdateEmbeddingBody(BaseModel):
    id: int
    image: UploadFile = File()

# Dependencies
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post('/onboard')
def onboard(files: list[UploadFile] = File(...)):
    embeddings = []

    for file in files:
        img = Image.open(file.file).convert("RGB")
        embeddings.append(model.forward(img))
    embeddings = convert_to_postgres(embeddings)
    item = put_embeddings(SessionLocal(), embeddings)
    # return item.id
    return {'message':item.id}

@app.post('/update_embeddings')
def update(id: int, image: UploadFile):
# def update(file:UpdateEmbeddingBody):
    # image = Image.open(file.image.file).convert("RGB")
    image = Image.open(image.file).convert("RGB")
    encoding = model.forward(image)
    encodings = read_one_embedding(SessionLocal(), id)
    encodings = pythonise_embeddings(encodings.encodings)
    encodings.append(encoding)
    encodings = convert_to_postgres(encodings)
    new_encodings = update_embeddings(SessionLocal(), id, encodings)
    if pythonise_embeddings(new_encodings.encodings) == encodings:
        return {'message':'successfull'}
    return {'message':'unsuccessfull'}

@app.post('/reonboard/')
def reonboard(id: int, images: list[UploadFile] = File(...)):
    embeddings = []
    for image in images:
        image = Image.open(image.file).convert("RGB")
        embeddings.append(model.forward(image))

    embeddings = convert_to_postgres(embeddings)
    new_embeddings = update_embeddings(SessionLocal(), id, embeddings)
    if pythonise_embeddings(new_embeddings.encodings) == embeddings:
        return {'message':'successfull'}
    return {'message':'unsuccessfull'}
