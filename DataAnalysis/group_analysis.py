import keypoint_moseq as kpms
import matplotlib
import pandas as pd
import numpy as np
import os
import h5py
import glob
import argparse

parser = argparse.ArgumentParser(description="Analyze behavioral data.")
parser.add_argument('--project_dir', type=str, required=True, help="Path to the KPMS project directory")
parser.add_argument('--model', type=str, required=True, help="Name of the trained model")

args = parser.parse_args()

project_dir = args.project_dir
model_name = args.model

kpms.interactive_group_setting(project_dir, model_name)

moseq_df = kpms.compute_moseq_df(project_dir, model_name, smooth_heading=True)

print(moseq_df.head())

stats_df = kpms.compute_stats_df(
    project_dir,
    model_name,
    moseq_df,
    min_frequency=0.005,
    groupby=["group", "name"],
    fps=30,
)


num_sylls = len(stats_df['syllable'].unique())
print(num_sylls)


kpms.plot_syll_stats_with_sem(
    stats_df,
    project_dir,
    model_name,
    plot_sig=False,
    thresh=2,
    stat="duration",
    order="stat",
    ctrl_group="Pilot",
    exp_group="Harmaline",
    figsize=(8, 4),
    groups=stats_df["group"].unique(),
);

normalize = "bigram"

trans_mats, usages, groups, syll_include = kpms.generate_transition_matrices(
    project_dir,
    model_name,
    normalize=normalize,
    min_frequency=0.005,
)

kpms.visualize_transition_bigram(
    project_dir,
    model_name,
    groups,
    trans_mats,
    syll_include,
    normalize=normalize,
    show_syllable_names=True,
)


kpms.plot_transition_graph_group(
    project_dir,
    model_name,
    groups,
    trans_mats,
    usages,
    syll_include,
    layout="circular",
    show_syllable_names=False,
)
