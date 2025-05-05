import keypoint_moseq as kpms
import matplotlib
import pandas as pd
import numpy as np
import os
import h5py
import glob
import argparse

matplotlib.use('Agg')


def parse_args():
    parser = argparse.ArgumentParser(description="KPMS pipeline")
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--data_dirs", type=str, nargs='+', required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=12,
                        help="latent dimention for the model update (default=12)")
    return parser.parse_args()

args = parse_args()

# project_dir = '/data/jverpeut/JoVE_FineMotor/KPMS_models/Harmaline_Plinko'
project_dir = args.project_dir
data_dirs = args.data_dirs

nodes = [
    "Upperarm", "Wrist", "Palmcenter",
    "Base1", "Base2", "Base3", "Base4",
    "Knuckle1", "Knuckle2", "Knuckle3", "Knuckle4",
    "Fingertip1", "Fingertip2", "Fingertip3", "Fingertip4"
]
skeleton = [
    ["Upperarm", "Wrist"],
    ["Wrist", "Palmcenter"],
    ["Palmcenter", "Base1"],
    ["Palmcenter", "Base2"],
    ["Palmcenter", "Base3"],
    ["Palmcenter", "Base4"],
    ["Base1", "Knuckle1"],
    ["Base2", "Knuckle2"],
    ["Base3", "Knuckle3"],
    ["Base4", "Knuckle4"],
    ["Knuckle1", "Fingertip1"],
    ["Knuckle2", "Fingertip2"],
    ["Knuckle3", "Fingertip3"],
    ["Knuckle4", "Fingertip4"]
]


# data_path = "/data/jverpeut/JoVE_FineMotor/FineMotorTaskData/Harmaline/Plinko"

# h5_files = glob.glob(os.path.join(data_path, "*.h5"))
# csv_files = glob.glob(os.path.join(data_path, "*_bouts.csv"))

all_h5_files = []
all_csv_files = []

for i in args.data_dirs:
    h5_files = glob.glob(os.path.join(i, "*.h5"))
    csv_files = glob.glob(os.path.join(i, "*_bouts.csv"))

    all_h5_files.extend(h5_files)
    all_csv_files.extend(csv_files)

map_bouts = {
    os.path.basename(f).replace(".analysis_bouts.csv", ""): f
    for f in all_csv_files
}

print(all_csv_files)
print(all_h5_files)
print(list(map_bouts.keys()))

sleap_h5_files = []
bout_csv_files = []

for i in all_h5_files:
    basename = os.path.basename(i).replace(".analysis.h5", "")
    if basename in map_bouts:
        sleap_h5_files.append(i)
        bout_csv_files.append(map_bouts[basename])

if not sleap_h5_files:
    print("no files found")
    exit(1)


init_file = sleap_h5_files[0]

try:
    kpms.setup_project(project_dir, sleap_file=init_file, nodes=nodes, skeleton=skeleton, overwrite=True)
    print(f"Project directory: {project_dir}")
except:
    print(f"loading existing model")

config = lambda: kpms.load_config(project_dir)


kpms.update_config(project_dir,
                   anterior_bodyparts=["Fingertip1", "Fingertip2", "Fingertip3", "Fingertip4"],
                   posterior_bodyparts=["Upperarm"],
                   use_bodyparts=nodes,
                   skeleton=skeleton)

def get_frames_to_keep(bout_csv, total_frames):
    intervals_df = pd.read_csv(bout_csv)
    all_frames = []
    for _, row in intervals_df.iterrows():
        start_frame = int(row['bout_start'])
        end_frame = int(row['bout_end'])
        if start_frame < end_frame and start_frame < total_frames:
            end_frame = min(end_frame, total_frames)
            frames = list(range(start_frame, end_frame))
            all_frames.extend(frames)

    all_frames = sorted(set(all_frames))
    return all_frames

data_dict = {}
conf_dict = {}
found_bodyparts = None

for i, (sleap_file, bout_file) in enumerate(zip(sleap_h5_files, bout_csv_files)):
    print(f"{i+1}/{len(sleap_h5_files)}: {sleap_file}")
    print(f"bout intervals from {bout_file}")

    raw_data_dict, raw_conf_dict, raw_bodyparts = kpms.io.load_keypoints(sleap_file, 'sleap')

    if found_bodyparts is None:
        found_bodyparts = raw_bodyparts

    file_key = list(raw_data_dict.keys())[0]

    total_frames = raw_data_dict[file_key].shape[0]
    print(f"{total_frames}")

    frames_to_keep = get_frames_to_keep(bout_file, total_frames)
    print(f"Keeping {len(frames_to_keep)}")

    if len(frames_to_keep) == 0:
        print(f" Skipping {sleap_file} because it has no valid frames.")
        continue

    trimmed_data = raw_data_dict[file_key][frames_to_keep]
    trimmed_conf = raw_conf_dict[file_key][frames_to_keep]

    data_dict[file_key] = trimmed_data
    conf_dict[file_key] = trimmed_conf


if len(data_dict) == 0:
    exit(1)

recording_names = {}
for key in list(data_dict.keys()):
    if "Pilot" in key:
        recording_names[key] = "Pilot"
    elif "Harmaline" in key:
        recording_names[key] = "Harmaline"
    else:
        recording_names[key] = key

recordings = {}
recordings_conf = {}
for key, rec_name in recording_names.items():
    recordings[rec_name] = data_dict[key]
    recordings_conf[rec_name] = conf_dict[key]

print(f"{len(recordings)} recordings: {list(recordings.keys())}")

data, metadata = kpms.format_data(recordings, recordings_conf, **config())

pca = kpms.fit_pca(Y=data['Y'], mask=data['mask'], **config())
kpms.save_pca(pca, project_dir)
kpms.print_dims_to_explain_variance(pca, 0.9)
kpms.plot_scree(pca, project_dir=project_dir)
kpms.plot_pcs(pca, project_dir=project_dir, **config())

# perhaps break here and make two scripts, so that the proper latent_dim can be chosen for each batch
# resolved w/ argparse

kpms.update_config(project_dir, latent_dim=args.latent_dim)
model = kpms.init_model(data, pca=pca, **config())

# use 50 epochs as a minimum, 200-500 maximum to avoid overfitting.
epochs = args.epochs

print(f"Training model (AR-only phase for {epochs} epochs)")
model, model_name = kpms.fit_model(model, data, metadata, project_dir, ar_only=True, num_iters=epochs)
model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name, iteration=epochs)

model = kpms.update_hypparams(model, kappa=1e4)
model = kpms.fit_model(model, data, metadata, project_dir, model_name, ar_only=False,
                       start_iter=current_iter, num_iters=current_iter + 500)[0]

kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
results = kpms.extract_results(model, metadata, project_dir, model_name)
kpms.save_results_as_csv(results, project_dir, model_name)

print(f"Model ({model_name}) saved to {project_dir}")
