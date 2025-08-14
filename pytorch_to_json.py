import torch
import json

paths = ["./pt/pts.pt", "./pt/pts_embb.pt", "./pt/pts_embb_t.pt", "./pt/rgb.pt", "./pt/xyz.pt"]

for path in paths:
    tensor = torch.load(path, map_location=torch.device('cpu'))
    with open(path.replace(".pt", ".json"), "w") as f:
        json.dump(tensor.tolist(), f)