import os

import torch
from datauri import DataURI
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from predict import predict

UPLOAD_DIRECTORY = "/tmp/uploads"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

api = Flask(__name__)
CORS(api)


@api.route("/files")
def list_files():
    """Endpoint to list files on the server."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return jsonify(files)


@api.route("/files/<path:path>")
def get_file(path):
    """Download a file."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


@api.route("/classify_image", methods=["POST"])
def classify_image():
    """Classify an image"""

    url = request.json['url']
    uri = DataURI(url)

    tmp_image_path = UPLOAD_DIRECTORY + '/temp_image.png'
    with open(tmp_image_path, 'wb') as f:
        f.write(uri.data)

    model = torch.jit.load('model.pt')
    probs, classes = predict(tmp_image_path, model, topk=1, category_names='cat_to_name.json')

    # Return 201 CREATED
    return jsonify(
        {
            'class': probs[0],
            'confidence_score': classes[0]
        }
    )


if __name__ == "__main__":
    api.run(debug=True, port=8000)
