import keypoint_moseq as kpms
import matplotlib
import pandas as pd
import numpy as np
import os
import h5py
import glob

isset = False

project_dir = ''
model_name = '2025_04_02-23_19_54'

kpms.interactive_group_setting(project_dir, model_name)

moseq_df = kpms.compute_moseq_df(project_dir, model_name, smooth_heading=True)

print(moseq_df.head())

if isset:
    stats_df = kpms.compute_stats_df(
        project_dir,
        model_name,
        moseq_df,
        min_frequency=0.005,  # threshold frequency for including a syllable in the dataframe
        groupby=["group", "name"],  # column(s) to group the dataframe by
        fps=30,
    )  # frame rate of the video from which keypoints were inferred


    num_sylls = len(stats_df['syllable'].unique())
    print(num_sylls)


    kpms.plot_syll_stats_with_sem(
        stats_df,
        project_dir,
        model_name,
        plot_sig=False,  # whether to mark statistical significance with a star
        thresh=2,  # significance threshold
        stat="duration",  # statistic to be plotted (e.g. 'duration' or 'velocity_px_s_mean')
        order="stat",  # order syllables by overall frequency ("stat") or degree of difference ("diff")
        ctrl_group="Pilot",  # name of the control group for statistical testing
        exp_group="Harmaline",  # name of the experimental group for statistical testing
        figsize=(8, 4),  # figure size
        groups=stats_df["group"].unique(),  # groups to be plotted
    );



    normalize = "bigram"  # normalization method ("bigram", "rows" or "columns")

    trans_mats, usages, groups, syll_include = kpms.generate_transition_matrices(
        project_dir,
        model_name,
        normalize=normalize,
        min_frequency=0.005,  # minimum syllable frequency to include
    )

    kpms.visualize_transition_bigram(
        project_dir,
        model_name,
        groups,
        trans_mats,
        syll_include,
        normalize=normalize,
        show_syllable_names=True,  # label syllables by index (False) or index and name (True)
    )


    kpms.plot_transition_graph_group(
        project_dir,
        model_name,
        groups,
        trans_mats,
        usages,
        syll_include,
        layout="circular",  # transition graph layout ("circular" or "spring")
        show_syllable_names=False,  # label syllables by index (False) or index and name (True)
    )
