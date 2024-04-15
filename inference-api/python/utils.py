import torch
def pythonise_embeddings(encodings):
    l = []
    for i in encodings:
        k = []
        for j in i:
            n = []
            for m in j:
                # k.append(torch.tensor(float(m)))
                n.append(float(m))
            k.append(n)
        l.append(torch.tensor(k))
    return l

def convert_to_postgres(out):
    l = []
    for i in out:
        l.append(i.tolist())
    return l
