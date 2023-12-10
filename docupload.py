from waitress import serve
import logging
from formocr import FormOCR
from flask import Flask, request, jsonify
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_cors import CORS  # Import CORS from flask_cors

app = Flask(__name__)
analyzer = FormOCR()
CORS(app)

# Configuration for file uploads
photos = UploadSet("photos", IMAGES)
app.config["UPLOADED_PHOTOS_DEST"] = "uploads"

configure_uploads(app, photos)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "photo" not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files["photo"]

    if file.filename == "":
        return jsonify({"message": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = photos.save(file)
        result = analyzer.analyze(filename)
        return result
    else:
        return jsonify({"message": "File type not allowed"}), 400
        
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg", "gif"}

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    serve(app, host="0.0.0.0", port=8080)
