import argparse
from glob import glob
import re
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main(args):
    plotting_data = []
    for filename in glob("VQVAE/results/celeba_recon_wave_sweep-*-*-*/metrics*.txt"):
        match = re.match(r"VQVAE/results/celeba_recon_wave_sweep-(\d*\.?\d+)-(\d*\.?\d+)-(\d+)", filename)
        if match:
            w0, w1, level = match.group(1), match.group(2), match.group(3)
            with open(filename) as f:
                lines = [line.split(":") for line in f.read().strip().splitlines()]
                metrics = {metric.strip(): float(score.strip()) for metric, score in lines}
                for metric, score in metrics.items():
                    plotting_data.append({
                        "metric": metric,
                        "score": score,
                        "w0": w0,
                        "w1": w1,
                        "level": level
                    })

    df = pd.DataFrame(plotting_data)

    metrics = pd.unique(df["metric"]) if args.metrics is None else args.metrics
    levels = sorted(pd.unique(df["level"]))

    n_metrics = len(metrics)
    n_levels = len(levels)

    fig = make_subplots(
        rows=n_metrics, cols=n_levels,
        subplot_titles=[f"{metric} - Level {level}" for metric in metrics for level in levels],
        shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.1, horizontal_spacing=0.1
    )

    row = 1
    col = 1

    for metric in metrics:
        for level in levels:
            sub_df = df[(df["metric"] == metric.upper()) & (df["level"] == level)]
            
            pivoted = sub_df.pivot(index="w1", columns="w0", values="score")
            
            z = pivoted.values
            x = pivoted.columns
            y = pivoted.index
            
            heatmap_data = go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale='Viridis',
                colorbar=dict(title="Score"),
                showscale=False,
                text=z,
                texttemplate="%{text:.3f}"
            )

            
            fig.add_trace(
                heatmap_data,
                row=row, col=col
            )
            
            col += 1
        row += 1
        col = 1

    fig.update_layout(
        title="Heatmaps of Metrics for Different Loss Weights",
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_title="Weight 0 (w0)",
        yaxis_title="Weight 1 (w1)",
        xaxis=dict(title="Weight 0 (w0)"),
        yaxis=dict(title="Weight 1 (w1)")
    )

    fig.show()

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="*", help="Metrics to plot", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)