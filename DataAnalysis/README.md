# Keypoint-MoSeq Analysis Pipeline

Analysis pipeline for group comparison using keypoint-MoSeq.

## Usage

1. Fit keypoint-MoSeq Model

```sh
python run_kpms.py --project_dir /path/to/project --data_dirs /path/to/data1 /path/to/data2 --epochs <epochs> --latent_dim <latent_dim>
```

**Arguments**
- `--project_dir`: Output directory for KPMS project initialization
- `--data_dirs`: Input directories with SLEAP H5 tracking files and CSVs of behavioral bouts
- `--epochs` (default: 100): Training epochs for autoregressive phase
- `--latent_dim` (default: 12): Number of principal components used to represent pose trajectories

**Input files**
- `*.h5`: SLEAP tracking data
- `*_bouts.csv`: Behavioral bout intervals with `bout_start` and `bout_end` columns

Trains keypoint-MoSeq's AR-HMM to identify behavioral syllables in animal pose tracking data.

**Project structure**
```
data/
├── group1/
│   ├── session1.analysis.h5
│   ├── session1.analysis_bouts.csv
│   └── ...
└── group2/
    ├── session1.analysis.h5
    ├── session1.analysis_bouts.csv
    └── ...

videos/
├── group1/
│   ├── session1.avi
│   └── ...
└── group2/
    ├── session1.avi
    └── ...
```

2. Group Analysis

```bash
python group_analysis.py --project_dir /path/to/project --model model_name
```

Generates syllable statistics, syllable transition matrices, and syllable transition graphs to compare groups. 

3. Grid Movies

```bash
python grid_movie_analysis.py --project_dir /path/to/project --model model_name --video_dirs /path/to/videos1 /path/to/videos2
```

**Arguments:**
- `--project_dir`: Path to the KPMS project directory  
- `--model`: Name of the trained model
- `--video_dirs`: Directories containing AVI video files 

Creates video grids showing representative examples of each behavioral syllable.
