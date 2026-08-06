#This handles the LLM summary
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
openai_client = OpenAI()


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
    '''Generate LLM summary of model results'''
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