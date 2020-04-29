from flask import Flask, request, jsonify
from PIL import Image
from Services.NewIdentitiesManager import NewIdentitiesManager

app = Flask(__name__)
manager: NewIdentitiesManager = NewIdentitiesManager.get_manager('./locals')


@app.route('/register/<string:location>/<string:name>', methods=['POST'])
def register_identities(location, name):
    files = request.files
    images = [Image.open(request.files.get(file)) for file in files]
    manager.publish_identity(location, name, images[:-1], images[-1:])
    return jsonify(success=True)