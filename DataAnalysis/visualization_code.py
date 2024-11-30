import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv2
import kaleido
import plotly
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import csv
from kpms_analysis import *

plt.rcParams['pdf.fonttype'] = 42  # To work with in Illustrator
plt.rcParams['ps.fonttype'] = 42   # To work with in Illustrator

class BoutReader:
    def __init__(self, filepath, bout_gap, min_bout_length):
        self.filepath = filepath
        self.bout_gap = bout_gap
        self.min_bout_length = min_bout_length
        # self.data = []
        self.data = None
        self.bouts = []

    def _load_data(self):
        self.data = pd.read_csv(self.filepath)

    def find_bouts(self):
        present_node = ~self.data.iloc[:,1:].isna().any(axis=1) # must change is nans are not used for gaps

        gap_interval_start = None

        bout_interval_start = None

        current_gap_length = 0

        for i, j in enumerate(present_node):
            if j:

                if bout_interval_start is None:

                    bout_interval_start = i
                current_gap_length = 0
            else:

                if bout_interval_start is not None:
                    current_gap_length += 1 # to account for multiple gaps

                    if current_gap_length > self.bout_gap:

                        if i - bout_interval_start - current_gap_length >= self.min_bout_length:

                            self.bouts.append((bout_interval_start, i - current_gap_length))

                        bout_interval_start = None

                        current_gap_length = 0

        if bout_interval_start is not None:
            bout_interval_end = len(present_node)

            if bout_interval_end - bout_interval_start >= self.min_bout_length:
                self.bouts.append((bout_interval_start, bout_interval_end))

    def load_bouts(self):
        return self.bouts

    def write_bouts(self, output_filepath):

        with open(output_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['bout_start', 'bout_end'])

            for i, j in self.bouts: # start, end
                writer.writerow([i,j])

    def run(self):
        self._load_data()
        self.find_bouts()

class PlaceROI:
    def __init__(self, image):
        self.image = image

        self.points = []
        self.lines = []

        # ipynb placement
        self.output = widgets.Output()

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)
        self.print_coords = widgets.Label(value="Coordinates")
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.full_polygon = False



    def on_move(self, event):
      if event.inaxes:
        x, y = event.xdata, event.ydata
        self.print_coords.value = f"{x},{y}"

    def on_click(self, event):
        with self.output:
          if event.button == 1 and not self.full_polygon:
              x, y = event.xdata, event.ydata

              if len(self.points) > 0:
                  x1, y1 = self.points[0]

                  # compute distance for connections
                  if np.sqrt((x-x1)**2 + (y-y1)**2) < 10:  # make this value larger for more detailed ROIs
                      self.completed_ROI()

                      return
              self.points.append((x, y))

              self.ax.plot(x, y, 'ro')
              if len(self.points) > 1:

                  x0, y0 = self.points[-2]
                  line, = self.ax.plot([x0, x], [y0, y], 'r-')

                  self.lines.append(line)

              self.fig.canvas.draw()

    def completed_ROI(self):
        with self.output:
          if len(self.points) > 2:

              # connect the final and in initial points
              x1, y1 = self.points[0]
              xf, yf = self.points[-1]

              line, = self.ax.plot([xf, x1], [yf, y1], 'r-')

              self.lines.append(line)

              self.full_polygon = True

              self.fig.canvas.draw()

              print("ROI completed")

              # print immediatley for ipynbs
              for i, (x,y) in enumerate(self.points, start=1):
                print(f"point {i}: {x},{y}")

    def run(self):
        # plt.show()
        display(widgets.VBox([self.print_coords, self.output]))
        plt.show()

        # display(widgets.VBox([self.fig.canvas, self.print_coords, self.output]))


    def get_ROI_data(self):

        # just replace this and save data to a variable for a notebook

        if self.full_polygon:

            data = [(x, y) for x, y in self.points]
            return data

        else:
            return None

class ReadVideo:
    def __init__(self, filepath):
        self.filepath = filepath

    def get_first_frame(self):

        video = cv2.VideoCapture(self.filepath)
        ret, frame = video.read()

        video.release()

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

class LoadTrackingFile:

    def __init__(self, filepath, roi, nodes_to_track):
        self.filepath = filepath
        self.polygon = roi
        self.tracking_file_type = self._find_file_type()

        self.nodes = []

        self.positions = None
        self.nodes_to_track = nodes_to_track


    def _find_file_type(self):

        if self.filepath.endswith('.csv'):
            return 'csv'

        elif self.filepath.endswith('.h5'):
            return 'h5'

    def choose_load(self):
        if self.tracking_file_type == 'csv':
            self._load_csv()
        elif self.tracking_file_type == 'h5':
            self._load_h5()

    def _load_csv(self):
        data = pd.read_csv(self.filepath)
        self.nodes = [col.split('.')[0] for col in data.columns if col.endswith('.x')]

        frames = data['frame_idx'].max()+1
        track = data['track'].nunique()

        self.positions = np.zeros((track, 2, len(self.nodes), frames))

        for i,j in data.groupby('track'):
            track_number = int(i.split('_')[1])
            for k, m in enumerate(self.nodes):
                node_x = f'{m}.x'
                node_y = f'{m}.y'

                if node_x in j.columns and node_y in j.columns:
                    indicies = j["frame_idx"].values
                    self.positions[track_number, 0, k, indicies] = j[node_x].values
                    self.positions[track_number, 1, k, indicies] = j[node_y].values

    def _load_h5(self):
        with h5py.File(self.filepath, 'r') as h5:
            self.nodes = [node.decode() for node in h5['node_names'][:]]
            self.positions = h5["tracks"][:]

    def point_in_ROI(self, x, y):
        inside = np.zeros_like(x, dtype=bool)
        vx = np.asarray(x)
        vy = np.asarray(y)

        verticies = np.array(self.polygon)

        points = len(verticies)

        j = points -1

        for i in range(points):
            xi, yi = verticies[i]
            xj,yj = verticies[j]

            intersect = ((yi > y) != (yj > y)) & \
                        (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi)

            inside ^= intersect
            j = i

        return inside


    def write_output(self, output_filepath):
        if self.tracking_file_type == 'h5':
            frames = self.positions.shape[3]
            tracks = self.positions.shape[0]

        else:

            # frames = len(self.positions)
            # tracks = 1

            frames = self.positions.shape[3]
            tracks = self.positions.shape[0]

        with open(output_filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            header = ['frame'] + [f"{node}_x,{node}_y" for node in self.nodes_to_track]
            writer.writerow(header)

            for i in range(frames):
                row = [i]
                all_nodes_in_roi = True
                node_data = {}

                for j in range(tracks):
                    for idx, node in enumerate(self.nodes):
                        if node in self.nodes_to_track:
                            # if self.tracking_file_type == 'h5':
                                # x, y = self.positions[j, :, idx, i]
                            x,y = self.positions[j,0,idx,i], self.positions[j,1,idx,i]
                            # else:
                                # x,y = self.positions[i, idx * 2:(idx+1) * 2]

                            if self.point_in_ROI(x,y):
                                node_data[node] = [x,y]
                            else:
                                all_nodes_in_roi = False
                                break

                if all_nodes_in_roi and len(node_data) == len(self.nodes_to_track):
                    for node in self.nodes_to_track:
                        row.extend(node_data[node])
                else:
                    row.extend([np.nan, np.nan] * len(self.nodes_to_track))

                writer.writerow(row)

class DataPipelineAndBoxplot:
    def __init__(self, roi_data, nodes_to_track, frame_rate, mm_per_pixel, frame_threshold):
        # set parameters defined in the execution class
        self.roi_data = roi_data
        self.nodes_to_track = nodes_to_track
        self.frame_rate = frame_rate
        self.mm_per_pixel = mm_per_pixel
        self.frame_threshold = frame_threshold

        self.colors = {'Pilot': 'grey', 'Harmaline': 'black'}

        self.pilot_median_velocities = {node: [] for node in self.nodes_to_track}
        self.harmaline_median_velocities = {node: [] for node in self.nodes_to_track}

    def collect_h5_files(self):
        dir_pilot = 'path_to_pilot_tracking_files'
        dir_harmaline = 'path_to_harmaline_tracking_files'

        # aggregate h5 files
        self.pilot_h5_files = [os.path.join(dir_pilot, f) for f in os.listdir(dir_pilot) if f.endswith('.h5')]
        self.harmaline_h5_files = [os.path.join(dir_harmaline, f) for f in os.listdir(dir_harmaline) if f.endswith('.h5')]

    def process_files(self, h5_files, group, median_velocities_dict):
        for h5_file in h5_files:
            output_csv = h5_file.replace('.h5', '_node_intersections.csv')

            # use bout formation classes on all files
            processor = LoadTrackingFile(h5_file, self.roi_data, self.nodes_to_track)
            processor.choose_load()
            processor.write_output(output_csv)

            # compute bouts
            bout_reader = BoutReader(output_csv, bout_gap=5, min_bout_length=10)
            bout_reader.run()
            bouts_output_csv = h5_file.replace('.h5', '_bouts.csv')
            bout_reader.write_bouts(bouts_output_csv)

            # compute median velocities
            movement_plot = self.ComputeVelocities(
                bouts_output_csv,
                h5_file,
                frame_rate=self.frame_rate,
                mm_per_pixel=self.mm_per_pixel,
                group=group,
                nodes=self.nodes_to_track
            )

            median_velocities = movement_plot.compute_filtered_median_velocities(self.frame_threshold)

            # Append median velocities
            for node in self.nodes_to_track:
                median_velocities_dict[node].extend(median_velocities[node])

    def create_boxplots(self):
        box_data = []
        categories = []

        for node in self.nodes_to_track:
            categories.extend([f'Pilot {node}', f'Harmaline {node}'])

        for node in self.nodes_to_track:
            # pilot
            x_pilot = f'Pilot {node}'
            y_pilot = self.pilot_median_velocities[node]

            box_data.append(go.Box(
                y=y_pilot,
                name=x_pilot,
                marker=dict(color=self.colors['Pilot']),
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))

            # harmaline
            x_harmaline = f'Harmaline {node}'
            y_harmaline = self.harmaline_median_velocities[node]

            box_data.append(go.Box(
                y=y_harmaline,
                name=x_harmaline,
                marker=dict(color=self.colors['Harmaline']),
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))

        fig = go.Figure(data=box_data)
        fig.update_layout(
            title="Median Velocity Comparison",
            yaxis_title="Median Velocity (mm/s)",
            xaxis_title="Group and Node",

            width=800,
            height=600,

            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,

            margin=dict(l=70, r=50, t=70, b=70),
            font=dict(family="Arial", size=12),

            xaxis=dict(
                showline=True,
                linecolor='black',
                linewidth=1,
                mirror=True,
                tickangle=45,
                categoryorder='array',
                categoryarray=categories
            ),

            yaxis=dict(
                showline=True,
                linecolor='black',
                linewidth=1,
                mirror=True
            )
        )

        # export to pdf
        fig.write_image('median_velocity_comparison.pdf', format='pdf')

    def run(self):
        self.collect_h5_files()
        self.process_files(self.pilot_h5_files, 'Pilot', self.pilot_median_velocities)
        self.process_files(self.harmaline_h5_files, 'Harmaline', self.harmaline_median_velocities)
        self.create_boxplots()

    class ComputeVelocities:
        def __init__(self, csv_file, tracking_filepath, frame_rate, mm_per_pixel, group, nodes):
            self.bouts = pd.read_csv(csv_file)
            self._load_h5(tracking_filepath)
            self.frame_rate = frame_rate
            self.mm_per_pixel = mm_per_pixel
            self.group = group
            self.nodes = nodes

        def _load_h5(self, filepath):
            with h5py.File(filepath, 'r') as f:
                self.node_names = [node.decode() if isinstance(node, bytes) else node for node in f['node_names'][:]]
                self.node_data = f["tracks"][:]

        def compute_filtered_median_velocities(self, frame_threshold, bout_indices=None):
            median_velocities_per_node = {node: [] for node in self.nodes}

            if bout_indices is None:
                bout_indices = self.bouts.index.tolist()

            for bout_index in bout_indices:
                row = self.bouts.iloc[bout_index]

                if 'bout_start' in self.bouts.columns and 'bout_end' in self.bouts.columns:
                    s = int(row['bout_start'])
                    e = int(row['bout_end'])
                elif 'start_frame' in self.bouts.columns and 'end_frame' in self.bouts.columns:
                    s = int(row['start_frame'])
                    e = int(row['end_frame'])
                else:
                    # or skip processing
                    continue

                # find the number of frames in each
                # bout and cut off at threshold
                num_frames = e - s + 1
                if num_frames < frame_threshold:
                    continue

                velocities = {node: [] for node in self.nodes}


                # start a loop and iterate over each frame
                # compute velocities of each node
                for i in range(s, e):
                    frame = i
                    next_frame = frame + 1
                    # check bounds
                    if next_frame >= self.node_data.shape[3]:
                        break
                    for node in self.nodes:

                        # verify the nodes are in the data
                        if node in self.node_names:
                            idx_node = self.node_names.index(node)

                            # compute velocity between a frame and the next
                            x1 = self.node_data[0, 0, idx_node, frame]
                            y1 = self.node_data[0, 1, idx_node, frame]
                            x2 = self.node_data[0, 0, idx_node, next_frame]
                            y2 = self.node_data[0, 1, idx_node, next_frame]

                            # x and y displacement
                            dx = x2 - x1
                            dy = y2 - y1
                            dt = 1 / self.frame_rate

                            # find the speed and apply the pixel-mm conversion
                            speed = (np.sqrt(dx ** 2 + dy ** 2) * self.mm_per_pixel) / dt
                            velocities[node].append(speed)

                # the median velocity for each node
                for node in self.nodes:
                    if velocities[node]:
                        median_velocity = np.nanmedian(velocities[node])
                        median_velocities_per_node[node].append(median_velocity)

            return median_velocities_per_node

class MeanVelocityLinePlot:
    def __init__(self, csv_pilot, h5_pilot, csv_harmaline, h5_harmaline,
                 bout_index_pilot, bout_index_harmaline,
                 frame_rate, mm_per_pixel):

        self.csv_pilot = csv_pilot
        self.h5_pilot = h5_pilot
        self.csv_harmaline = csv_harmaline
        self.h5_harmaline = h5_harmaline
        self.bout_index_pilot = bout_index_pilot
        self.bout_index_harmaline = bout_index_harmaline
        self.frame_rate = frame_rate
        self.mm_per_pixel = mm_per_pixel
        self.nodes = ["Palmcenter", "Wrist", "Upperarm"]

        # parse down plot
        self.cut_time = 0.325
        self.start_time = 0.05

    def fill_missing(self, Y, kind="linear"):
        initial_shape = Y.shape
        Y = Y.reshape((initial_shape[0], -1))
        for i in range(Y.shape[-1]):
            y = Y[:, i]
            x = np.flatnonzero(~np.isnan(y))
            if len(x) > 1:
                f = interp1d(x, y[x], kind=kind, fill_value="extrapolate", bounds_error=False)
                y_filled = f(np.arange(len(y)))
                Y[:, i] = y_filled
            elif len(x) == 1:
                y[:] = y[x[0]]
                Y[:, i] = y
        Y = Y.reshape(initial_shape)
        return Y

    def load_h5(self, filepath):
        with h5py.File(filepath, 'r') as f:
            node_names = [node.decode() for node in f['node_names'][:]]
            node_data = f["tracks"][:]
            node_data = self.fill_missing(node_data, kind="linear")
        return node_names, node_data

    def compute_distance(self, bouts, node_names, node_data, bout_index):
        row = bouts.iloc[bout_index]

        s = int(row['bout_start']) if 'bout_start' in row else int(row['start_frame'])
        e = int(row['bout_end']) if 'bout_end' in row else int(row['end_frame'])

        mean_point_distances = []
        median_point_distances = []

        for i in range(s, e):
            frame = i
            next_frame = frame + 1
            # check bouts
            if next_frame >= node_data.shape[3]:
                mean_point_distances.append(np.nan)
                median_point_distances.append(np.nan)
                continue
            per_frame_distances = []
            # verify the nodes are in the data
            for node in self.nodes:
                if node in node_names:
                    idx_node = node_names.index(node)

                    # compute velocity between a frame and the next
                    x1, y1 = node_data[0, 0, idx_node, frame], node_data[0, 1, idx_node, frame]
                    x2, y2 = node_data[0, 0, idx_node, next_frame], node_data[0, 1, idx_node, next_frame]

                    # x and y displacement
                    dx, dy = x2 - x1, y2 - y1

                    # find the speed and apply the pixel-mm conversion
                    distance = np.sqrt(dx**2 + dy**2) * self.mm_per_pixel
                    per_frame_distances.append(distance)

            # check for valid distances
            if per_frame_distances:
                mean_distance = np.mean(per_frame_distances)
                median_distance = np.median(per_frame_distances)
                mean_point_distances.append(mean_distance)
                median_point_distances.append(median_distance)
            else:
                mean_point_distances.append(np.nan)
                median_point_distances.append(np.nan)

        # interpolation
        mean_point_distances = pd.Series(mean_point_distances).interpolate(method='linear', limit_direction='both').to_numpy()
        median_point_distances = pd.Series(median_point_distances).interpolate(method='linear', limit_direction='both').to_numpy()
        return mean_point_distances, median_point_distances

    def calculate_auc(self, time, distance):
        return simps(distance, time)

    def plot_and_save(self, time_pilot_cut, pilot_distances_cut, time_harmaline_cut, harmaline_distances_cut, plot_title, filename):
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=time_pilot_cut,
            y=pilot_distances_cut,
            mode='lines',
            name='Pilot',
            line=dict(color='black'),
            connectgaps=True
        ))
        fig_line.add_trace(go.Scatter(
            x=time_harmaline_cut,
            y=harmaline_distances_cut,
            mode='lines',
            name='Harmaline',
            line=dict(color='dodgerblue'),
            connectgaps=True
        ))
        fig_line.update_layout(
            title={'text': plot_title, 'x': 0.5, 'xanchor': 'center'},
            xaxis_title="Time (s)",
            yaxis_title="Distance (mm)",
            width=800,
            height=600,
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend=dict(
                x=1.02,
                y=1,
                xanchor='left',
                yanchor='top',
                orientation='v',
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(l=50, r=150, t=50, b=50),
            xaxis=dict(
                range=[0, self.cut_time - self.start_time],
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True
            ),
            yaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black',
                rangemode='tozero'
            )
        )
        fig_line.write_image(filename, format='pdf')

    def run_analysis(self):
        # Load data
        bouts_pilot = pd.read_csv(self.csv_pilot)
        node_names_pilot, node_data_pilot = self.load_h5(self.h5_pilot)
        bouts_harmaline = pd.read_csv(self.csv_harmaline)
        node_names_harmaline, node_data_harmaline = self.load_h5(self.h5_harmaline)

        # compute distances
        pilot_mean_distances, pilot_median_distances = self.compute_distance(
            bouts_pilot, node_names_pilot, node_data_pilot, self.bout_index_pilot)
        # mean and median distrances traveled
        harmaline_mean_distances, harmaline_median_distances = self.compute_distance(
            bouts_harmaline, node_names_harmaline, node_data_harmaline, self.bout_index_harmaline)

        # time arrays for each group
        time_pilot = np.linspace(0, len(pilot_mean_distances) / self.frame_rate, len(pilot_mean_distances))
        time_harmaline = np.linspace(0, len(harmaline_mean_distances) / self.frame_rate, len(harmaline_mean_distances))

        # set a cut time and determine indicies for each group within the interval
        pilot_indices = (time_pilot >= self.start_time) & (time_pilot <= self.cut_time)
        harmaline_indices = (time_harmaline >= self.start_time) & (time_harmaline <= self.cut_time)

        time_pilot_cut = time_pilot[pilot_indices] - self.start_time
        pilot_mean_distances_cut = pilot_mean_distances[pilot_indices]
        pilot_median_distances_cut = pilot_median_distances[pilot_indices]

        time_harmaline_cut = time_harmaline[harmaline_indices] - self.start_time
        harmaline_mean_distances_cut = harmaline_mean_distances[harmaline_indices]
        harmaline_median_distances_cut = harmaline_median_distances[harmaline_indices]

        pilot_mean_distances_cut = pd.Series(pilot_mean_distances_cut).interpolate(method='linear', limit_direction='both').to_numpy()
        pilot_median_distances_cut = pd.Series(pilot_median_distances_cut).interpolate(method='linear', limit_direction='both').to_numpy()
        harmaline_mean_distances_cut = pd.Series(harmaline_mean_distances_cut).interpolate(method='linear', limit_direction='both').to_numpy()
        harmaline_median_distances_cut = pd.Series(harmaline_median_distances_cut).interpolate(method='linear', limit_direction='both').to_numpy()

        auc_pilot_mean = self.calculate_auc(time_pilot_cut, pilot_mean_distances_cut)
        auc_harmaline_mean = self.calculate_auc(time_harmaline_cut, harmaline_mean_distances_cut)

        auc_pilot_median = self.calculate_auc(time_pilot_cut, pilot_median_distances_cut)
        auc_harmaline_median = self.calculate_auc(time_harmaline_cut, harmaline_median_distances_cut)

        # Plotting
        self.plot_and_save(
            time_pilot_cut, pilot_mean_distances_cut,
            time_harmaline_cut, harmaline_mean_distances_cut,
            "Mean Point Distance Over Time for Specified Bouts (Mean)",
            '/content/mean_point_distance_specified_bouts_mean.pdf'
        )

        self.plot_and_save(
            time_pilot_cut, pilot_median_distances_cut,
            time_harmaline_cut, harmaline_median_distances_cut,
            "Mean Point Distance Over Time for Specified Bouts (Median)",
            '/content/mean_point_distance_specified_bouts_median.pdf'
        )

        # format csvs
        pilot_df = pd.DataFrame({
            'Time': time_pilot_cut,
            'Mean_Point_Distance': pilot_mean_distances_cut,
            'Median_Point_Distance': pilot_median_distances_cut,
            'Group': ['Pilot'] * len(time_pilot_cut)
        })

        harmaline_df = pd.DataFrame({
            'Time': time_harmaline_cut,
            'Mean_Point_Distance': harmaline_mean_distances_cut,
            'Median_Point_Distance': harmaline_median_distances_cut,
            'Group': ['Harmaline'] * len(time_harmaline_cut)
        })

        # save a data csv for external AUC analysis
        line_plot_data = pd.concat([pilot_df, harmaline_df], ignore_index=True)
        line_plot_data.to_csv('/content/line_plot_data.csv', index=False)

        # Bar plots
        self.plot_bar(
            ['Pilot', 'Harmaline'],
            [auc_pilot_mean, auc_harmaline_mean],
            "Comparison of AUC values for Pilot and Harmaline (Mean)",
            '/content/auc_bar_plot_mean.pdf'
        )

        self.plot_bar(
            ['Pilot', 'Harmaline'],
            [auc_pilot_median, auc_harmaline_median],
            "Comparison of AUC values for Pilot and Harmaline (Median)",
            '/content/auc_bar_plot_median.pdf'
        )

    def plot_bar(self, categories, auc_values, plot_title, filename):
        # use final colors
        colors = ['black', 'dodgerblue']
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=categories,
            y=auc_values,
            marker_color=colors,
            # remove
            text=[f"{val}" for val in auc_values],
            textposition='auto'
        ))
        fig_bar.update_layout(
            title={'text': plot_title, 'x': 0.5, 'xanchor': 'center'},
            xaxis_title="Group",
            yaxis_title="AUC (mm·s)",
            width=600,
            height=500,
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            margin=dict(l=50, r=50, t=100, b=50),
            xaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True
            ),
            yaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='black',
                rangemode='tozero'
            )
        )
        fig_bar.write_image(filename, format='pdf')

class RunKeypointMoseqAnalysis:
    @staticmethod
    def execute():
        project_dir = 'moseq-analysis'
        # specific to either group
        sleap_file = "pilot_or_harmaline.h5"
        intervals_csv_path = 'pilot_or_harmaline_bouts.csv'

        interpolation_type = 'scipy'

        nodes = [
            "Upperarm",
            "Wrist",
            "Palmcenter",
            "Base1",
            "Base2",
            "Base3",
            "Base4",
            "Knuckle1",
            "Knuckle2",
            "Knuckle3",
            "Knuckle4",
            "Fingertip1",
            "Fingertip2",
            "Fingertip3",
            "Fingertip4"
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

        analysis = MotionAnalysis(
            project_dir=project_dir,
            sleap_file=sleap_file,
            intervals_csv_path=intervals_csv_path,
            nodes=nodes,
            skeleton=skeleton,
            interpolation_type=interpolation_type
        )

        analysis.run()

class RunROIPlacer:
    @staticmethod
    def execute():
        # run to set ROI using the place ROI class
        video_dir = 'path_to_video_dir'

        Video_FILE = os.path.join(video_dir, r"path_to_video.avi")
        # check video frames per second

        video = cv2.VideoCapture(Video_FILE)
        frame_rate = video.get(cv2.CAP_PROP_FPS)
        print(f"Frame Rate: {frame_rate} fps")
        video.release()

        read_video = ReadVideo(Video_FILE)
        frame = read_video.get_first_frame()

        place_roi = PlaceROI(frame)

        place_roi.run()

        data = place_roi.get_ROI_data()

        if data:
            for i, j in enumerate(data):
                print(f"point {i+1}: ({j[0]}, {j[1]})")

class ComputeNodeIntersectionsAndBouts:
    @staticmethod
    def execute_intersections():
        nodes = [
            # example nodes
            "Wrist",
            "Palmcenter",
            "Upperarm"
        ]

        tracking_filepath = 'path_to_tracking_file.csv'
        roi_data = []  # Replace with ROI vertices (minimum of 4 points)

        processor = LoadTrackingFile(tracking_filepath, roi_data, nodes)

        processor.choose_load()

        if '/pilot/' in dir.lower():
            print("GROUP: pilot")
            processor.write_output("/content/node_intersections_pilot.csv")
        elif '/harmaline/' in dir.lower():
            print("GROUP: harmaline")
            processor.write_output("/content/node_intersections_harmaline.csv")

    @staticmethod
    def execute_bouts():
        if '/pilot/' in dir.lower():
            intesection_csv = "/content/node_intersections_pilot.csv"
            output = "/content/intersection_bouts_pilot.csv"
        elif '/harmaline/' in dir.lower():
            intesection_csv = "/content/node_intersections_harmaline.csv"
            output = "/content/intersection_bouts_harmaline.csv"

        bout_gap = 30  # the number of frames with no skeleton-ROI intersection allowed to be skipped for bout formation
        min_bout_length = 0

        reader = BoutReader(intesection_csv, bout_gap, min_bout_length)
        reader.run()
        reader.write_bouts(output)

class RunDataAggregationAndBoxPlot:
    @staticmethod
    def execute():
        # add ROI data
        # example parameters
        video_dir = 'path_to_video_dir'
        Video_FILE = os.path.join(video_dir, r"path_to_video.avi")
        # check video frames per second
        video = cv2.VideoCapture(Video_FILE)
        frame_rate = video.get(cv2.CAP_PROP_FPS)
        print(f"Frame Rate: {frame_rate} fps")
        video.release()
        roi_data = []
        nodes_to_track = ["Wrist", "Palmcenter", "Upperarm"]
        # use a milimeter to pixel conversion
        mm_per_pixel = 0.11969
        frame_threshold = 50  # Adjust as needed

        analysis = DataPipelineAndBoxplot(
            roi_data=roi_data,
            nodes_to_track=nodes_to_track,
            frame_rate=frame_rate,
            mm_per_pixel=mm_per_pixel,
            frame_threshold=frame_threshold
        )
        analysis.run()

class RunMeanVelocityLinePlot:
    @staticmethod
    def execute():
        # requires bouts and tracking files for both groups to run comparison
        # paths should be for one group
        csv_pilot = 'pilot_bouts.csv'
        h5_pilot = 'pilot_tracking.h5'
        csv_harmaline = 'harmaline_bouts.csv'
        h5_harmaline = 'harmaline_tracking.h5'
        # indicies from the bout csvs to compare
        bout_index_pilot = 1
        bout_index_harmaline = 2
        video_dir = 'path_to_video_dir'
        Video_FILE = os.path.join(video_dir, r"path_to_video.avi")
        # check video frames per second
        video = cv2.VideoCapture(Video_FILE)
        frame_rate = video.get(cv2.CAP_PROP_FPS)
        print(f"Frame Rate: {frame_rate} fps")
        video.release()
        # use a milimeter to pixel conversion
        mm_per_pixel = 0.11969

        analysis = MeanVelocityLinePlot(
            csv_pilot, h5_pilot, csv_harmaline, h5_harmaline,
            bout_index_pilot, bout_index_harmaline,
            frame_rate, mm_per_pixel
        )
        analysis.run_analysis()

if __name__ == "__main__":
    # to place ROIs on video
    RunROIPlacer.execute()

    # compute node intersections and form bouts of intersection from the placed ROI
    ComputeNodeIntersectionsAndBouts.execute_intersections()
    ComputeNodeIntersectionsAndBouts.execute_bouts()

    # run keypoint-moseq analysis
    RunKeypointMoseqAnalysis.execute()

    # run boxplot
    RunDataAggregationAndBoxPlot.execute()

    # run line plot
    RunMeanVelocityLinePlot.execute()
