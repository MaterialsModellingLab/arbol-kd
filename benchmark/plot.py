#!/usr/bin/env python3
# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0
"""Plot the results of the KD-Tree benchmark."""

import argparse
import json
import pathlib
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"  # Use the browser for rendering plots


def plot(data: dict) -> go.Figure:
    """Plot the benchmark results from a JSON file."""
    df = pd.json_normalize(data["benchmarks"])
    methods = df["name"].str.extract(r"^test_(.*?)(?:\[.*\])?$")
    df["method"] = methods

    fig = px.line(
        df,
        x="params.x",
        y="stats.mean",
        color="method",
        error_y=df["stats.stddev"],
        labels={
            "name": "Test Name",
            "stats.mean": "Mean Time (s)",
            "params.x": "Number of Points",
        },
    )
    fig.update_yaxes(type="log")
    fig.update_xaxes(type="log")
    fig.update_layout(
        legend={
            "title": "",
            "orientation": "h",
            "y": 1.02,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "bottom",
        }
    )
    return fig


def main(base_path: str, show: bool) -> None:
    """Main function to plot benchmark results."""
    # Recursive search for JSON files in the base path
    base_path = pathlib.Path(base_path)
    files = list(base_path.glob("**/*.json"))
    if not files:
        print("No JSON files found in the specified path.")
        sys.exit(1)
    file_map = {}
    files.sort(key=lambda x: x.stem)
    for f in files:
        _, stem = f.stem.split("_", 1)
        file_map[stem] = f

    for f in file_map.values():
        with open(f, encoding="utf-8") as file:
            data = json.load(file)
        fig = plot(data)

        if show:
            fig.show()
        _, file_stem = f.stem.split("_", 1)

        fig.write_image(
            file_stem + ".svg",
            width=800,
            height=600,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show the plot in the browser.",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=".benchmarks",
        help="Base path to search for JSON files.",
    )
    args = parser.parse_args()
    main(**vars(args))
