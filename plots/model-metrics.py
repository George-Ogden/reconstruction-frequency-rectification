import argparse
from glob import glob
import re
import pandas as pd
from plotly.subplots import make_subplots

import plotly.express as px


def main(args):
    plotting_data = []
    for filename in glob("VQVAE/results/model-sweep-*-*/metrics*.txt"):
        match = re.match(r"VQVAE/results/model-sweep-(resnet\d+)", filename)
        if match:
            model = match.group(1)
            with open(filename) as f:
                lines = [line.split(":") for line in f.read().strip().splitlines()]
                metrics = {metric.strip(): float(score.strip()) for metric, score in lines}
                for metric, score in metrics.items():
                    plotting_data.append({
                        "metric": metric,
                        "score": score,
                        "model": model,
                    })
    plotting_data.sort(key=lambda x: int(x["model"].lower().replace("resnet","")))

    df = pd.DataFrame(plotting_data)

    metrics = pd.unique(df["metric"]) if args.metrics is None else args.metrics
    for metric in metrics:
        sub_df = df[df["metric"] == metric.upper()]
        fig = px.box(sub_df, x="model", color="model", y="score")
        fig.update_layout(title=metric.title())

    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_yaxes=True,
        subplot_titles=[metric.title() for metric in metrics]
    )

    for i, metric in enumerate(metrics):
        sub_df = df[df["metric"] == metric.upper()]
        box_fig = px.box(sub_df, x="model", y="score", color="model")
        
        for trace in box_fig.data:
            fig.add_trace(trace, row=i + 1, col=1)

    fig.update_layout(
        title="Metrics Comparison Across Models",
        showlegend=False
    )
    fig.show()

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="*", help="Metrics to plot", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

