#Handles creating the charts and plots for OLS and bootstrap results

import plotly.graph_objects as go
import plotly.io as pio

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
