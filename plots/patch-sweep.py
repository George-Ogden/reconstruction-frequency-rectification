import argparse
from glob import glob
import re
import pandas as pd
from plotly.subplots import make_subplots

import plotly.express as px

def main(args):
    plotting_data = []

    for filename in glob("VQVAE/results/celeba_recon_model_sweep-*-*/metrics*.txt") + glob("VQVAE/results/*ffl/metrics*.txt"):
        match = re.match(r"VQVAE/results/.+((resnet\d+)|(j|w|(wo))_ffl)", filename)
        if match:
            model = match.group(1)
            with open(filename) as f:
                lines = [line.split(":") for line in f.read().strip().splitlines()]
                metrics = {metric.strip(): float(score.strip()) for metric, score in lines}
                for metric, score in metrics.items():
                    plotting_data.append({
                        "metric": metric,
                        "score": score,
                        "variant": model,
                    })

    for filename in glob("VQVAE/results/celeba_recon_ffl_patch_sweep*/metrics*.txt"):
        match = re.match(r"VQVAE/results/[^/]+?((\d+)|(j|w|(wo))_ffl)", filename)
        if match:
            patch = match.group(1)
            try:
                int(patch)
                patch = f"patch-{patch}"
            except ValueError:
                ...
            with open(filename) as f:
                lines = [line.split(":") for line in f.read().strip().splitlines()]
                metrics = {metric.strip(): float(score.strip()) for metric, score in lines}
                for metric, score in metrics.items():
                    plotting_data.append({
                        "metric": metric,
                        "score": score,
                        "patch": patch
                    })



    df = pd.DataFrame(plotting_data)
    print(df.pivot_table(values="score", index="patch", columns="metric"))

    metrics = pd.unique(df["metric"]) if args.metrics is None else args.metrics
    for metric in metrics:
        sub_df = df[df["metric"] == metric.upper()]
        fig = px.box(sub_df, x="patch", color="patch", y="score")
        fig.update_layout(title=metric.title())

    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_yaxes=True,
        subplot_titles=[metric.title() for metric in metrics]
    )

    for i, metric in enumerate(metrics):
        sub_df = df[df["metric"] == metric.upper()]
        box_fig = px.box(sub_df, x="patch", y="score", color="patch")
        
        for trace in box_fig.data:
            fig.add_trace(trace, row=i + 1, col=1)

    fig.update_layout(
        title="Metrics Comparison Across Patch Sizes",
        showlegend=False
    )
    # fig.show()

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="*", help="Metrics to plot", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

