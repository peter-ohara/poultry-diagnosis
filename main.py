import os

from flask import Flask, request, abort, jsonify, send_from_directory
from flask_cors import CORS
import random

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

    print(request.json)

    # Return 201 CREATED
    return jsonify(
        {
            'class': random.choice(['normal', 'abnormal']),
            'confidence_score': random.uniform(0.75, 0.92)
        }
    )


if __name__ == "__main__":
    api.run(debug=True, port=8000)
