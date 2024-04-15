from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
from database import Base, engine, SessionLocal, read_all_embeddings
from utils import pythonise_embeddings
from new_models import SiameseNetwork, Network

Base.metadata.create_all(engine)

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained = torch.load('./model1.pt', map_location=device)
# pretrained = torch.load('./final_training_model.pt', map_location=device)
siamese = SiameseNetwork()
siamese.load_state_dict(pretrained)
siamese.eval()

embeddings = {}
query_result = read_all_embeddings(SessionLocal())
for i in query_result:
    embeddings[i.id] = pythonise_embeddings(i.encodings)

model = Network(model=siamese, embeddings = embeddings)
model.eval()

@app.post('/inference')
def inference(images: list[UploadFile] = File(...)):
    l = []
    for image in images:
        img = Image.open(image.file).convert("RGB")
        out = model.forward(img)
        l.append(out)
    return {'result': l}
