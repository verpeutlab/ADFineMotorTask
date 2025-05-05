import os
import re
import glob
import keypoint_moseq as kpms

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

def extract_identifier(filename):
    m = re.search(r'(\d{12,})', filename)
    return m.group(1) if m else None

dirs = [
    "/data/jverpeut/JoVE_FineMotor/FineMotorTaskData/Harmaline_copy/Plain",
    "/data/jverpeut/JoVE_FineMotor/FineMotorTaskData/Pilot_copy/Plain"
]

avi_files = []
h5_files = []
for d in dirs:
    avi_files.extend(glob.glob(os.path.join(d, "*.avi")))
    h5_files.extend(glob.glob(os.path.join(d, "*.h5")))

avi_dict = {}
for avi in avi_files:
    ident = extract_identifier(os.path.basename(avi))
    if ident:
        avi_dict[ident] = avi

video_paths = {}
for h5 in h5_files:
    ident = extract_identifier(os.path.basename(h5))
    if ident and ident in avi_dict:
        name = os.path.basename(h5).split(".avi")[0] + ".avi"
        video_paths[name] = avi_dict[ident]

project_dir = "/data/jverpeut/JoVE_FineMotor/KPMS_models/Comparison_Plain"
model_name = "2025_04_04-23_19_28"
results = kpms.load_results(project_dir, model_name)
coordinates, confidences, bodyparts = kpms.load_keypoints(dirs, "sleap")

video_paths = {}
for key in results:
    ident = extract_identifier(key)
    if ident and ident in avi_dict:
        video_paths[key] = avi_dict[ident]

kpms.generate_grid_movies(
    results,
    project_dir=project_dir,
    model_name=model_name,
    coordinates=coordinates,
    video_paths=video_paths,
    min_frequency=0.005,
    keypoints_only=False,
    overlay_keypoints=True,
    use_bodyparts=nodes,
    skeleton=skeleton
)

config = lambda: kpms.load_config(project_dir)

kpms.generate_trajectory_plots(coordinates, results, project_dir, model_name, **config())
