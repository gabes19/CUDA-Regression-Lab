import os
import pandas as pd
import statsmodels.api as sm
from flask import Flask, render_template, request
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


@app.route("/analyze", methods=["POST"])
def analyze():
    filename = request.form.get("filename")
    research_question = request.form.get("research_question")
    dependent_variable = request.form.get("dependent_variable")
    main_independent_variable = request.form.get("main_independent_variable")
    controls = request.form.getlist("controls")
    bootstrap_iterations = request.form.get("bootstrap_iterations")

    csv_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    df = pd.read_csv(csv_path)

    model_results = fit_models(
        df = df,
        dependent_variable = dependent_variable,
        main_independent_variable= main_independent_variable,
        controls= controls
    )

    baseline_coefficient = model_results[0]["coefficient"]
    final_coefficient = model_results[-1]["coefficient"]
    coefficient_change = final_coefficient - baseline_coefficient

    return render_template(
        "results.html",
        research_question=research_question,
        dependent_variable=dependent_variable,
        main_independent_variable=main_independent_variable,
        controls=controls,
        bootstrap_iterations=bootstrap_iterations,
        models=model_results,
        baseline_coefficient=baseline_coefficient,
        final_coefficient=final_coefficient,
        coefficient_change=coefficient_change,
    )

def fit_models(df, dependent_variable, main_independent_variable, controls):
    '''Helper function to fit multiple models and return results'''
    y = df[dependent_variable]
    model_results = []
    for i in range(len(controls) +1):
        current_controls = controls[:i]
        x_columns = [main_independent_variable] + current_controls
        
        X = df[x_columns]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        model_results.append({
            "model_name": f"Model {i+1}",
            "formula": f"{dependent_variable} ~ " + " + ".join(x_columns),
            "controls": current_controls,
            "coefficient": float(model.params[main_independent_variable]),
            "p_value": float(model.pvalues[main_independent_variable]),
            "r_squared": float(model.rsquared),
            "n_observations": int(model.nobs),
        })

    return model_results
