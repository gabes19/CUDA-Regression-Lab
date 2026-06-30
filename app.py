import os
import re
import subprocess
import zipfile
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
from flask import Flask, abort, render_template, request, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from openai import OpenAI
import json
import uuid
from datetime import datetime

matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
openai_client = OpenAI()


app = Flask(__name__)

def clean_metric(value):
    '''Return JSON/template-friendly floats for model metrics.'''
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(metric):
        return None

    return metric

@app.template_filter("metric")
def format_metric(value, digits=3):
    metric = clean_metric(value)
    if metric is None:
        return "n/a"

    return f"{metric:.{digits}f}"

#Report export folder for local dev
REPORTS_FOLDER = "reports"
app.config["REPORTS_FOLDER"] = REPORTS_FOLDER

#Upload folder for local dev
UPLOAD_FOLDER = "uploads"
SAMPLE_DATASET_FILENAME = "wage_education_sample.csv"
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

@app.route("/sample/wage-education")
def sample_wage_education():
    '''Load the bundled wage/education sample dataset.'''
    sample_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        SAMPLE_DATASET_FILENAME
    )

    if not os.path.exists(sample_path):
        abort(404, description="Sample dataset not found")

    columns = parse_columns(sample_path)
    return render_template(
        "configure.html",
        filename=SAMPLE_DATASET_FILENAME,
        columns=columns
    )

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
    try:
        llm_summary = generate_llm_summary(llm_payload=llm_payload)
    except Exception as error:
        print(f"LLM summary failed: {error}")
        llm_summary = (
            "LLM summary unavailable. Review the coefficient table, coefficient stability chart, "
            "and bootstrap interval directly. This is an associational regression analysis, "
            "not causal proof."
        )
    export_payload = build_export_payload(
        research_question=research_question,
        dependent_variable=dependent_variable,
        main_independent_variable=main_independent_variable,
        controls=controls,
        bootstrap_iterations=bootstrap_iterations,
        model_results=model_results,
        baseline_coefficient=baseline_coefficient,
        final_coefficient=final_coefficient,
        coefficient_change=coefficient_change,
        coefficient_chart=coefficient_chart,
        bootstrap_results=bootstrap_results,
        llm_summary=llm_summary
    )
    export_token = store_export_payload(export_payload)

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
        llm_summary=llm_summary,
        export_token=export_token
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
        coefficient_interval = model.conf_int().loc[main_independent_variable]

        model_results.append({
            "model_name": f"Model {i+1}",
            "formula": f"{dependent_variable} ~ " + " + ".join(x_columns),
            "controls": current_controls,
            "coefficient": clean_metric(model.params[main_independent_variable]),
            "standard_error": clean_metric(model.bse[main_independent_variable]),
            "t_value": clean_metric(model.tvalues[main_independent_variable]),
            "p_value": clean_metric(model.pvalues[main_independent_variable]),
            "ci_95": [
                clean_metric(coefficient_interval[0]),
                clean_metric(coefficient_interval[1]),
            ],
            "r_squared": clean_metric(model.rsquared),
            "adjusted_r_squared": clean_metric(model.rsquared_adj),
            "rmse": clean_metric(np.sqrt(model.mse_resid)),
            "f_statistic": clean_metric(model.fvalue),
            "f_p_value": clean_metric(model.f_pvalue),
            "n_observations": int(model.nobs),
            "df_residual": clean_metric(model.df_resid),
            "condition_number": clean_metric(model.condition_number),
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

def create_coefficient_figure(coefficient_chart, main_independent_variable):
    '''Create a Plotly figure for coefficient stability across models.'''
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

    return fig

def create_coefficient_plot(coefficient_chart, main_independent_variable):
    '''Create a Plotly line chart for coefficient stability across models.'''
    fig = create_coefficient_figure(coefficient_chart, main_independent_variable)

    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

def create_bootstrap_histogram_figure(bootstrap_results, main_independent_variable):
    '''Create a Plotly figure for bootstrapped coefficient samples.'''
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

    return fig

def create_bootstrap_histogram_plot(bootstrap_results, main_independent_variable):
    '''Create a Plotly histogram for bootstrapped coefficient samples.'''
    fig = create_bootstrap_histogram_figure(
        bootstrap_results,
        main_independent_variable
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

def generate_llm_summary(llm_payload):
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        text={
            "verbosity":"low"
        },
        input=[
            {
                "role":"system",
                "content":(
                    "You summarize regression analysis results for students and researchers."
                    "Use only the structured results provided. Do not invent diagnostics, causality, "
                    "or facts about the raw dataset."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize these regression results in 4 concise bullet points. "
                    "Be objective, plain-English, and avoid speculation. "
                    "Each bullet must be one sentence. "
                    "Do not include long explanations, caveats beyond the required causal caveat, or methodological background. "
                    "Include only: main finding, robustness after controls, bootstrap uncertainty, and next check. "
                    f"{llm_payload}"
                ), 
            },
        ],
    )

    return response.output_text

def build_export_payload(
    research_question,
    dependent_variable,
    main_independent_variable,
    controls,
    bootstrap_iterations,
    model_results,
    baseline_coefficient,
    final_coefficient,
    coefficient_change,
    coefficient_chart,
    bootstrap_results,
    llm_summary,
):
    '''Build a compact report payload for immediate PDF/LaTeX export.'''
    return {
        "created_at": datetime.utcnow().isoformat(),
        "research_question": research_question,
        "dependent_variable": dependent_variable,
        "main_independent_variable": main_independent_variable,
        "controls": controls,
        "bootstrap_iterations": bootstrap_iterations,
        "models": model_results,
        "baseline_coefficient": baseline_coefficient,
        "final_coefficient": final_coefficient,
        "coefficient_change": coefficient_change,
        "coefficient_chart": coefficient_chart,
        "bootstrap_results": bootstrap_results,
        "llm_summary": llm_summary,
    }

def store_export_payload(export_payload):
    '''Store the current analysis payload for export downloads.'''
    export_token = uuid.uuid4().hex
    payload_dir = Path(app.config["REPORTS_FOLDER"]) / "export_payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)

    payload_path = payload_dir / f"{export_token}.json"
    with payload_path.open("w", encoding="utf-8") as file:
        json.dump(export_payload, file, indent=2)

    return export_token

def validate_export_token(export_token):
    if not re.fullmatch(r"[0-9a-f]{32}", export_token or ""):
        abort(404)

def load_export_payload(export_token):
    validate_export_token(export_token)
    payload_path = (
        Path(app.config["REPORTS_FOLDER"])
        / "export_payloads"
        / f"{export_token}.json"
    )

    if not payload_path.exists():
        abort(404, description="Export payload not found")

    with payload_path.open("r", encoding="utf-8") as file:
        return json.load(file)

def report_dir_for_token(export_token):
    validate_export_token(export_token)
    report_dir = Path(app.config["REPORTS_FOLDER"]) / export_token
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir

def latex_escape(value):
    '''Escape text for safe use in a LaTeX document.'''
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    text = "" if value is None else str(value)
    return "".join(replacements.get(character, character) for character in text)

def format_number(value, digits=3):
    metric = clean_metric(value)
    if metric is None:
        return "n/a"

    return f"{metric:.{digits}f}"

def latex_summary(summary):
    lines = [line.strip() for line in str(summary or "").splitlines()]
    lines = [line for line in lines if line]

    if not lines:
        return latex_escape(
            "No LLM summary was generated for this analysis. "
            "This is an associational regression analysis, not causal proof."
        )

    return "\n\n".join(latex_escape(line) for line in lines)

def write_report_graphs(payload, report_dir):
    coefficient_image = report_dir / "coefficient_stability.png"
    bootstrap_image = report_dir / "bootstrap_histogram.png"

    model_names = [point["model_name"] for point in payload["coefficient_chart"]]
    coefficients = [point["coefficient"] for point in payload["coefficient_chart"]]
    main_variable = payload["main_independent_variable"]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(model_names, coefficients, marker="o", color="#111111", linewidth=2)
    ax.axhline(0, color="#999999", linewidth=1, linestyle="--")
    ax.set_title("Coefficient Stability")
    ax.set_ylabel(f"{main_variable} coefficient")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(coefficient_image, dpi=180)
    plt.close(fig)

    bootstrap = payload["bootstrap_results"]
    samples = bootstrap["samples"]
    ci_lower, ci_upper = bootstrap["ci_95"]
    mean = bootstrap["mean"]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.hist(samples, bins=28, color="#333333", edgecolor="#ffffff")
    ax.axvline(mean, color="#111111", linewidth=2, label="Mean")
    ax.axvline(ci_lower, color="#777777", linewidth=1.5, linestyle="--", label="95% CI")
    ax.axvline(ci_upper, color="#777777", linewidth=1.5, linestyle="--")
    ax.set_title("Bootstrap Distribution")
    ax.set_xlabel(f"Bootstrapped {main_variable} coefficient")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(bootstrap_image, dpi=180)
    plt.close(fig)

    return coefficient_image, bootstrap_image

def build_latex_document(payload):
    controls = payload.get("controls") or []
    controls_text = ", ".join(controls) if controls else "None"
    bootstrap = payload["bootstrap_results"]
    ci_lower, ci_upper = bootstrap["ci_95"]

    model_rows = []
    for model in payload["models"]:
        ci_lower, ci_upper = model.get("ci_95") or [None, None]
        ci_text = f"{format_number(ci_lower, 3)} to {format_number(ci_upper, 3)}"

        model_rows.append(
            " & ".join([
                latex_escape(model.get("model_name")),
                latex_escape(model.get("formula")),
                format_number(model.get("coefficient"), 4),
                format_number(model.get("standard_error"), 4),
                format_number(model.get("t_value"), 3),
                format_number(model.get("p_value"), 4),
                latex_escape(ci_text),
                format_number(model.get("r_squared"), 4),
                format_number(model.get("adjusted_r_squared"), 4),
                format_number(model.get("rmse"), 3),
                format_number(model.get("f_statistic"), 3),
                format_number(model.get("f_p_value"), 4),
                str(model.get("n_observations", "n/a")),
            ]) + r" \\"
        )

    model_table = "\n".join(model_rows)

    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage[T1]{{fontenc}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.7em}}

\begin{{document}}

\begin{{center}}
{{\Large CUDA Regression Lab Report}}\\
\vspace{{0.25em}}
{{\small Generated {latex_escape(payload["created_at"])}}}
\end{{center}}

\section*{{Research Question}}
{latex_escape(payload["research_question"])}

\section*{{Model Setup}}
\textbf{{Dependent variable:}} {latex_escape(payload["dependent_variable"])}\\
\textbf{{Main independent variable:}} {latex_escape(payload["main_independent_variable"])}\\
\textbf{{Controls:}} {latex_escape(controls_text)}

\section*{{Main Results}}
\textbf{{Baseline coefficient:}} {format_number(payload["baseline_coefficient"])}\\
\textbf{{Final coefficient:}} {format_number(payload["final_coefficient"])}\\
\textbf{{Coefficient change:}} {format_number(payload["coefficient_change"])}

\section*{{Coefficient Stability}}
\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{coefficient_stability.png}}
\end{{figure}}

\section*{{Bootstrap Uncertainty}}
\textbf{{Iterations:}} {payload["bootstrap_iterations"]}\\
\textbf{{Mean coefficient:}} {format_number(bootstrap["mean"])}\\
\textbf{{Standard error:}} {format_number(bootstrap["standard_error"])}\\
\textbf{{95\% interval:}} {format_number(ci_lower)} to {format_number(ci_upper)}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{bootstrap_histogram.png}}
\end{{figure}}

\section*{{Model Progression}}
\scriptsize
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrlrrrrrr}}
\hline
Model & Formula & Coef. & SE & T & P & 95\% CI & R-sq & Adj. R-sq & RMSE & F & F P & N \\
\hline
{model_table}
\hline
\end{{tabular}}
}}
\normalsize

\section*{{LLM Research Summary}}
{latex_summary(payload["llm_summary"])}

\vfill
\textit{{This is an associational regression analysis, not causal proof.}}

\end{{document}}
"""

def ensure_report_artifacts(export_token):
    payload = load_export_payload(export_token)
    report_dir = report_dir_for_token(export_token)

    try:
        write_report_graphs(payload, report_dir)
    except RuntimeError as error:
        abort(500, description=str(error))

    tex_path = report_dir / "regression_report.tex"
    tex_path.write_text(build_latex_document(payload), encoding="utf-8")
    return report_dir, tex_path

def build_latex_zip(report_dir, tex_path):
    zip_path = report_dir / "regression_report_latex.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(tex_path, arcname=tex_path.name)
        archive.write(
            report_dir / "coefficient_stability.png",
            arcname="coefficient_stability.png"
        )
        archive.write(
            report_dir / "bootstrap_histogram.png",
            arcname="bootstrap_histogram.png"
        )
    return zip_path

def compile_pdf_report(report_dir, tex_path):
    try:
        result = subprocess.run(
            [
                "pdflatex",
                "-enable-installer",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ],
            cwd=report_dir,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError:
        abort(
            500,
            description=(
                "pdflatex was not found. Install a LaTeX distribution such as "
                "MiKTeX or TeX Live to enable PDF export."
            ),
        )
    except subprocess.TimeoutExpired:
        abort(500, description="pdflatex timed out while compiling the report.")

    if result.returncode != 0:
        log_tail = (result.stdout + result.stderr)[-1600:]
        abort(500, description=f"pdflatex failed:\n{log_tail}")

    pdf_path = report_dir / "regression_report.pdf"
    if not pdf_path.exists():
        abort(500, description="pdflatex finished but did not create a PDF.")

    return pdf_path

@app.route("/export/latex/<export_token>")
def export_latex(export_token):
    report_dir, tex_path = ensure_report_artifacts(export_token)
    zip_path = build_latex_zip(report_dir, tex_path)

    return send_file(
        zip_path,
        as_attachment=True,
        download_name="cuda_regression_lab_latex.zip",
        mimetype="application/zip",
    )

@app.route("/export/pdf/<export_token>")
def export_pdf(export_token):
    report_dir, tex_path = ensure_report_artifacts(export_token)
    pdf_path = compile_pdf_report(report_dir, tex_path)

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name="cuda_regression_lab_report.pdf",
        mimetype="application/pdf",
    )
