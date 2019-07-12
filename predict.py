#!/usr/bin/env python

import argparse
import json

import torch
from PIL import Image

from helper import process_image


def predict(image_path, model, topk=2, category_names='cat_to_name.json', device="cpu"):
    """ Predict the class (or classes) of a flower image using a trained deep learning model.
    """
    image = Image.open(image_path)
    image = process_image(image).unsqueeze(dim=0)
    image = image.type(torch.FloatTensor)

    image = image.to(device)
    model = model.to(device)

    logps = model(image)
    ps = torch.exp(logps)
    top_ps, top_class = ps.topk(topk, dim=1)

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    idx_to_class = {v: k for k, v in {'newcastle': 0, 'normal': 1}.items()}

    if topk > 1:
        top_ps = [top_p.item() for top_p in top_ps.squeeze()]
        top_class = [idx_to_class[top_class.item()] for top_class in top_class.squeeze()]
    else:
        top_ps = [top_p.item() for top_p in top_ps]
        top_class = [idx_to_class[top_class.item()] for top_class in top_class]

    top_class_name = [cat_to_name[top_class] for top_class in top_class]

    return top_class_name, top_ps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("image_file", help="path to image of flower")
    parser.add_argument("checkpoint",
                        help="path to torch checkpoint containing trained flower classification model")
    parser.add_argument("--top_k", help="return top KK most likely classes", type=int, default=2)
    parser.add_argument("--category_names", help="use a mapping of categories to real names",
                        default='cat_to_name.json')
    parser.add_argument("--gpu", help="Use GPU for inference", action="store_true")

    args = parser.parse_args()

    device = "cuda" if args.gpu else "cpu"
    model = torch.jit.load('model.pt')
    probs, classes = predict(args.image_file, model, topk=args.top_k, category_names=args.category_names, device=device)
    print(probs)
    print(classes)
