import os

from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename


app = Flask(__name__)

#Upload folder for local dev
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def start():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    '''Handles user CSV upload'''
    uploaded_file = request.files.get("csv_file")

    if uploaded_file is None or uploaded_file.filename == "":
        return "No file uploaded", 400
    
    if not uploaded_file.filename.endswith(".csv"):
        return "Please upload a CSV file", 400
    
    filename = secure_filename(uploaded_file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    uploaded_file.save(save_path)

    return f"Uploaded {filename} succesffuly"
