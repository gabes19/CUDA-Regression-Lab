import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
openai_client = OpenAI()


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

    bootstrap_iterations = int(bootstrap_iterations)
    bootstrap_results = bootstrap_coefficient(
        df=df,
        dependent_variable=dependent_variable,
        main_independent_variable=main_independent_variable,
        controls=controls,
        iterations=bootstrap_iterations,
        )
    baseline_coefficient = model_results[0]["coefficient"]
    final_coefficient = model_results[-1]["coefficient"]
    coefficient_change = final_coefficient - baseline_coefficient
    coefficient_chart = create_coefficient_chart(model_results)
    coefficient_plot_html = create_coefficient_plot(
        coefficient_chart,
        main_independent_variable
    )
    bootstrap_histogram_html = create_bootstrap_histogram_plot(
        bootstrap_results,
        main_independent_variable
    )
    llm_payload = build_llm_summary_payload(
    research_question=research_question,
    dependent_variable=dependent_variable,
    main_independent_variable=main_independent_variable,
    controls=controls,
    model_results=model_results,
    bootstrap_results=bootstrap_results,
    bootstrap_iterations=bootstrap_iterations,
    baseline_coefficient=baseline_coefficient,
    final_coefficient=final_coefficient,
    coefficient_change=coefficient_change,
)

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
        coefficient_chart=coefficient_chart,
        coefficient_plot_html=coefficient_plot_html,
        bootstrap_results=bootstrap_results,
        bootstrap_histogram_html=bootstrap_histogram_html,
    )

def fit_models(df, dependent_variable, main_independent_variable, controls):
    '''Fit multiple models and return results'''
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

def create_coefficient_chart(model_results):
    '''Create a chart for model coefficient results'''
    coefficient_chart = []
    for model in model_results:
        coefficient_chart.append({
            "model_name": model["model_name"],
            "coefficient": model["coefficient"],
        })
    return coefficient_chart

def create_coefficient_plot(coefficient_chart, main_independent_variable):
    '''Create a Plotly line chart for coefficient stability across models.'''
    model_names = [point["model_name"] for point in coefficient_chart]
    coefficients = [point["coefficient"] for point in coefficient_chart]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=model_names,
        y=coefficients,
        mode="lines+markers",
        line={
            "color": "#f4f4f4",
            "width": 2,
        },
        marker={
            "color": "#070707",
            "line": {
                "color": "#f4f4f4",
                "width": 2,
            },
            "size": 9,
        },
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{main_independent_variable} coefficient: "
            "%{y:.4f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=None,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="#0b0b0b",
        font={
            "family": "Courier New, monospace",
            "color": "#f4f4f4",
        },
        margin={
            "l": 54,
            "r": 24,
            "t": 24,
            "b": 46,
        },
        height=320,
        xaxis={
            "title": None,
            "gridcolor": "#303030",
            "linecolor": "#555555",
            "tickfont": {"color": "#b7b7b7"},
            "zeroline": False,
        },
        yaxis={
            "title": f"{main_independent_variable} coefficient",
            "gridcolor": "#303030",
            "linecolor": "#555555",
            "tickfont": {"color": "#b7b7b7"},
            "zeroline": True,
            "zerolinecolor": "#555555",
        },
        hoverlabel={
            "bgcolor": "#151515",
            "bordercolor": "#555555",
            "font": {
                "family": "Courier New, monospace",
                "color": "#f4f4f4",
            },
        },
    )

    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

def create_bootstrap_histogram_plot(bootstrap_results, main_independent_variable):
    '''Create a Plotly histogram for bootstrapped coefficient samples.'''
    samples = bootstrap_results["samples"]
    ci_lower, ci_upper = bootstrap_results["ci_95"]
    mean = bootstrap_results["mean"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=samples,
        nbinsx=28,
        marker={
            "color": "#f4f4f4",
            "line": {
                "color": "#0b0b0b",
                "width": 1,
            },
        },
        opacity=0.88,
        hovertemplate=(
            "Coefficient range: %{x}<br>"
            "Count: %{y}<extra></extra>"
        ),
    ))

    fig.add_vline(
        x=mean,
        line_color="#f4f4f4",
        line_width=2,
        line_dash="solid",
        annotation_text="mean",
        annotation_font_color="#f4f4f4",
    )
    fig.add_vline(
        x=ci_lower,
        line_color="#b7b7b7",
        line_width=1,
        line_dash="dash",
        annotation_text="2.5%",
        annotation_font_color="#b7b7b7",
    )
    fig.add_vline(
        x=ci_upper,
        line_color="#b7b7b7",
        line_width=1,
        line_dash="dash",
        annotation_text="97.5%",
        annotation_font_color="#b7b7b7",
    )

    fig.update_layout(
        title=None,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="#0b0b0b",
        bargap=0.06,
        font={
            "family": "Courier New, monospace",
            "color": "#f4f4f4",
        },
        margin={
            "l": 54,
            "r": 24,
            "t": 24,
            "b": 52,
        },
        height=320,
        xaxis={
            "title": f"Bootstrapped {main_independent_variable} coefficient",
            "gridcolor": "#303030",
            "linecolor": "#555555",
            "tickfont": {"color": "#b7b7b7"},
            "zeroline": False,
        },
        yaxis={
            "title": "Count",
            "gridcolor": "#303030",
            "linecolor": "#555555",
            "tickfont": {"color": "#b7b7b7"},
            "zeroline": False,
        },
        hoverlabel={
            "bgcolor": "#151515",
            "bordercolor": "#555555",
            "font": {
                "family": "Courier New, monospace",
                "color": "#f4f4f4",
            },
        },
    )

    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

def bootstrap_coefficient(df, dependent_variable, main_independent_variable, controls, iterations):
    '''Bootstrap the main coefficient from the full model'''
    x_columns = [main_independent_variable] + controls
    coefficients = []
    for _ in range(iterations):
        sample_df = df.sample(
            n=len(df),
            replace=True
        )

        y = sample_df[dependent_variable]
        X = sample_df[x_columns]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        coefficients.append(float(model.params[main_independent_variable]))

    coefficients = np.array(coefficients)

    return {
        "mean": float(np.mean(coefficients)),
        "standard_error": float(np.std(coefficients, ddof=1)),
        "ci_95": [
            float(np.percentile(coefficients, 2.5)),
            float(np.percentile(coefficients, 97.5)),
        ],
        "samples": coefficients.tolist(),
    }

def build_llm_summary_payload(
        research_question,
        dependent_variable,
        main_independent_variable,
        controls,
        model_results,
        bootstrap_results,
        bootstrap_iterations,
        baseline_coefficient,
        final_coefficient,
        coefficient_change,
):
     return {
        "research_question": research_question,
        "dependent_variable": dependent_variable,
        "main_independent_variable": main_independent_variable,
        "controls": controls,
        "model_progression": model_results,
        "coefficient_summary": {
            "baseline_coefficient": baseline_coefficient,
            "final_coefficient": final_coefficient,
            "coefficient_change": coefficient_change,
        },
        "bootstrap": {
            "iterations": bootstrap_iterations,
            "mean": bootstrap_results["mean"],
            "standard_error": bootstrap_results["standard_error"],
            "ci_95": bootstrap_results["ci_95"],
        },
        "diagnostics_warnings": [],
    }


