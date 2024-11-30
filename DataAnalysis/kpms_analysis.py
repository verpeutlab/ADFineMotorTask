import os
import keypoint_moseq as kpms
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import shutil

class KeypointMoseqAnalysis:
    def __init__(self, project_dir, sleap_file, intervals_csv_path, nodes, skeleton, interpolation_type='scipy'):
        self.project_dir = project_dir
        self.sleap_file = sleap_file
        self.intervals_csv_path = intervals_csv_path
        self.nodes = nodes
        self.skeleton = skeleton
        self.interpolation_type = interpolation_type

    def setup_project(self):
        if os.path.exists(self.project_dir):
            shutil.rmtree(self.project_dir)
        kpms.setup_project(self.project_dir, sleap_file=self.sleap_file, nodes=self.nodes, skeleton=self.skeleton)

    def config(self):
        return kpms.load_config(self.project_dir)

    # linear interpolation
    def fill_missing(self, Y, kind="linear"):
        initial_shape = Y.shape
        Y = Y.reshape((initial_shape[0], -1))
        for i in range(Y.shape[-1]):
            y = Y[:, i]
            x = np.flatnonzero(~np.isnan(y))
            if len(x) > 1:
                f = interp1d(x, y[x], kind=kind, fill_value="extrapolate", bounds_error=False)
                xq = np.flatnonzero(np.isnan(y))
                y[xq] = f(xq)
                mask = np.isnan(y)
                y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])
            Y[:, i] = y
        return Y.reshape(initial_shape)

    def load_data(self):
        coordinates, confidences, found_bodyparts = kpms.io.load_keypoints(self.sleap_file, 'sleap')

        if set(found_bodyparts) != set(self.nodes):
            found_bodyparts = self.nodes

        kpms.update_config(
            self.project_dir,

            anterior_bodyparts=["Fingertip1", "Fingertip2", "Fingertip3", "Fingertip4"],
            posterior_bodyparts=["Upperarm"],
            # anterior_bodyparts = ["Upperarm"],
            # posterior_bodyparts = ["Fingertip1", "Fingertip2", "Fingertip3", "Fingertip4"],

            use_bodyparts=self.nodes,
            skeleton=self.skeleton
        )

        if self.interpolation_type == 'scipy':
            for key in coordinates:
                coordinates[key] = self.fill_missing(coordinates[key])
        # else:
        #     pass

        for key in coordinates:
            if np.isnan(coordinates[key]).any():
                print(f"nans for {key}")
                nan_mask = ~np.isnan(coordinates[key]).any(axis=(1, 2))

                coordinates[key] = coordinates[key][nan_mask]
                confidences[key] = confidences[key][nan_mask]

        self.coordinates = coordinates
        self.confidences = confidences

    def process_intervals(self):
        intervals_df = pd.read_csv(self.intervals_csv_path)

        bouts_coordinates = {key: [] for key in self.coordinates.keys()}
        bouts_confidences = {key: [] for key in self.confidences.keys()}

        # format bouts
        for _, row in intervals_df.iterrows():

            start_frame = int(row['bout_start'])
            end_frame = int(row['bout_end'])

            if start_frame >= end_frame:
                continue

            for key in self.coordinates:

                # cuts bouts
                max_frame = self.coordinates[key].shape[0]

                if end_frame > max_frame:
                    end_frame = max_frame

                bouts_coordinates[key].append(self.coordinates[key][start_frame:end_frame])
                bouts_confidences[key].append(self.confidences[key][start_frame:end_frame])

        for key in bouts_coordinates:
            if bouts_coordinates[key]:

                bouts_coordinates[key] = np.concatenate(bouts_coordinates[key], axis=0)
                bouts_confidences[key] = np.concatenate(bouts_confidences[key], axis=0)


                total_frames = bouts_coordinates[key].shape[0]

                # check for nans
                nan_frames = np.isnan(bouts_coordinates[key]).any(axis=(1, 2))
                missing_data = nan_frames.mean()

        self.bouts_coordinates = bouts_coordinates
        self.bouts_confidences = bouts_confidences

    def format_data(self):
        data, metadata = kpms.format_data(self.bouts_coordinates, self.bouts_confidences, **self.config())

        missing_data = {}
        for idx, bp in enumerate(self.nodes):
            nan_frames = np.isnan(data['Y'][:, idx, :]).any(axis=1)
            missing_data[bp] = nan_frames.mean()
            if missing_data[bp] > 0.5:
                print(f"significant amount of data missing for {bp}")

        threshold = 0.5
        valid_bodyparts = [bp for bp in self.nodes if missing_data[bp] <= threshold]

        # remove nodes with very little data
        if len(valid_bodyparts) < len(self.nodes):

            data, metadata = kpms.format_data(
                self.bouts_coordinates, self.bouts_confidences,
                use_bodyparts=valid_bodyparts, **self.config())
            kpms.update_config(self.project_dir, use_bodyparts=valid_bodyparts)

        self.data = data
        self.metadata = metadata

    def run_pca(self):
        pca = kpms.fit_pca(Y=self.data['Y'], mask=self.data['mask'], **self.config())
        kpms.save_pca(pca, self.project_dir)

        # dimensions to explain 90% of variance
        kpms.print_dims_to_explain_variance(pca, 0.9)

        # scree plot
        kpms.plot_scree(pca, project_dir=self.project_dir)

        kpms.plot_pcs(pca, project_dir=self.project_dir, **self.config())

        self.pca = pca

    def train_model(self):
        kpms.update_config(self.project_dir, latent_dim=4)


        model = kpms.init_model(self.data, pca=self.pca, **self.config())
        epochs = 50

        # autoregressive training for 50 epochs
        model, model_name = kpms.fit_model(
            model, self.data, self.metadata, self.project_dir, ar_only=True, num_iters=epochs)

        model, data, metadata, current_iter = kpms.load_checkpoint(
            self.project_dir, model_name, iteration=epochs)
        model = kpms.update_hypparams(model, kappa=1e4)

        # non-autoregressive training for 50 + 500 epochs
        model = kpms.fit_model(model, data, metadata, self.project_dir, model_name, ar_only=False, start_iter=current_iter, num_iters=current_iter + 500)[0]

        # reindex before plotting
        kpms.reindex_syllables_in_checkpoint(self.project_dir, model_name)

        self.model = model
        self.model_name = model_name

    def plot_results(self):
        # extract results and plot
        results = kpms.plot_results(self.model, self.metadata, self.project_dir, self.model_name)

        kpms.save_results_as_csv(results, self.project_dir, self.model_name)

        config_params = self.config()
        config_params.pop('project_dir', None)

        kpms.plot_syllable_frequencies(
            results, project_dir=self.project_dir, model_name=self.model_name, **config_params)

        kpms.generate_trajectory_plots(
            self.bouts_coordinates, results, project_dir=self.project_dir, model_name=self.model_name, **self.config())

        kpms.generate_grid_movies(
            results, project_dir=self.project_dir, model_name=self.model_name, coordinates=self.bouts_coordinates, **self.config())

        kpms.plot_similarity_dendrogram(
            self.bouts_coordinates, results, project_dir=self.project_dir, model_name=self.model_name, **self.config())

    def run(self):
        self.setup_project()
        self.load_data()
        self.process_intervals()
        self.format_data()
        self.run_pca()
        self.train_model()
        self.plot_results()
