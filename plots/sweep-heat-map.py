import argparse
from glob import glob
import re
import pandas as pd

import plotly.express as px


def main(args):
    plotting_data = []
    for filename in glob("VQVAE/results/coarse-weight-sweep-*-*/metrics*.txt"):
        match = re.match(r"VQVAE/results/coarse-weight-sweep-(\d*\.?\d+)-(\d*\.?\d+)", filename)
        if match:
            w0, w1 = match.group(1), match.group(2)
            with open(filename) as f:
                lines = [line.split(":") for line in f.read().strip().splitlines()]
                metrics = {metric.strip(): float(score.strip()) for metric, score in lines}
                for metric, score in metrics.items():
                    plotting_data.append({
                        "metric": metric,
                        "score": score,
                        "w0": w0,
                        "w1": w1
                    })

    df = pd.DataFrame(plotting_data)

    for metric in (pd.unique(df["metric"]) if args.metrics is None else args.metrics):
        sub_df = df[df["metric"] == metric.upper()]
        fig = px.imshow(sub_df.pivot(index="w1", columns="w0", values="score"), 
                        text_auto=True, color_continuous_scale="Viridis")
        fig.update_layout(title=f"Heatmap of {metric} for different loss weights")
        fig.show()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="*", help="Metrics to plot", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)