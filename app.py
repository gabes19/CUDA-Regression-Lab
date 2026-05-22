import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pandas as pd


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

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    uploaded_file.save(save_path)
    columns = parse_columns(save_path)

    return render_template("configure.html", filename=filename, columns=columns)

def parse_columns(csv_path):
    '''Helper function to parse column metadata for
    user to choose target variables and prepare for LLM'''
    df = pd.read_csv(csv_path)
    column_metadata = []
    for column in df.columns:
        column_metadata.append({
            "name": column,
            "dtype": str(df[column].dtype),
            "missing_values": int(df[column].isna().sum()),
            "unique_values": int(df[column].nunique())
        })
    return column_metadata
