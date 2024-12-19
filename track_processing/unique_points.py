import pandas as pd

import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import splrep, splev
import numpy as np

def fit_spline_closed(x, y, z, s=0):
    """Fit a closed spline to the given x, y, z coordinates."""
    # Parameterize the data
    t = np.linspace(0, 1, len(x))

    # Fit splines for x, y, z as a function of parameter t
    spline_x = splrep(t, x, s=s, per=True)
    spline_y = splrep(t, y, s=s, per=True)
    spline_z = splrep(t, z, s=s, per=True)

    return spline_x, spline_y, spline_z

def generate_spline_points(spline_x, spline_y, spline_z, num_points=100):
    """Generate evenly spaced points along the closed spline."""
    # Create an array of evenly spaced parameter values t
    t_new = np.linspace(0, 1, num_points)

    # Evaluate spline at the new t values
    x_new = splev(t_new, spline_x)
    y_new = splev(t_new, spline_y)
    z_new = splev(t_new, spline_z)

    return np.column_stack((x_new, y_new, z_new))

def get_evenly_spaced_points_on_spline(df, labels=['x', 'y', 'z'], num_points=2000):
        # Drop rows with any NaN values in the specified columns
        df = df.dropna(subset=labels)
        x = df[labels[0]].values
        y = df[labels[1]].values
        z = df[labels[2]].values

        # Fit spline to the data
        spline_x, spline_y, spline_z = fit_spline_closed(x, y, z)

        # Generate new points along the spline
        new_points = generate_spline_points(spline_x, spline_y, spline_z, num_points)

        if np.array_equal(new_points[0,:], new_points[-1,:]):
            new_points = np.vstack((new_points, new_points[0,:]))

        # Create a new DataFrame with the new points
        new_df = pd.DataFrame({
            labels[0]: new_points[:,0],
            labels[1]: new_points[:,1],
            labels[2]: new_points[:,2]
        })

        return new_df

def get_center_line_from_bounds(df):
    left_df = get_evenly_spaced_points_on_spline(df, ['left_bound_x','left_bound_y','left_bound_z'], 1000)
    right_df = get_evenly_spaced_points_on_spline(df, ['right_bound_x','right_bound_y','right_bound_z'], 1000)
    center_df = pd.concat([left_df, right_df], axis=1)
    center_df['x_m'] = (center_df['left_bound_x'] + center_df['right_bound_x']) / 2
    center_df['y_m'] = (center_df['left_bound_y'] + center_df['right_bound_y']) / 2
    # center_df['center_z'] = (center_df['left_bound_z'] + center_df['right_bound_z']) / 2
    center_df['w_tr_right_m'] = np.sqrt((center_df['right_bound_x'] - center_df['x_m'])**2 + (center_df['right_bound_y'] - center_df['y_m'])**2)
    center_df['w_tr_left_m'] = np.sqrt((center_df['left_bound_x'] - center_df['x_m'])**2 + (center_df['left_bound_y'] - center_df['y_m'])**2)
    center_df['banking_rad'] = np.zeros(center_df['x_m'].shape)
    return center_df

def plot_track(df, plot_name):
    fig, ax = plt.subplots()
    ax.plot(df['left_bound_x'], df['left_bound_y'], 'r')
    ax.plot(df['right_bound_x'], df['right_bound_y'], 'b')
    # ax.plot(df['x_m'], df['y_m'], 'g')
    ax.axis('equal')
    fig.size = (200, 200)
    ax.set_title(plot_name)
    # plt.show()

def main():
    source_file = Path('~/tri_workspace/online_3D_racing_line_planning/data/raw_track_data/thunderhill_east_raw.csv').expanduser()
    source_dir = source_file.parent
    raw_df = pd.read_csv(str(source_file))
    trimmed_df = raw_df
    trimmed_df['left_bound_x'] = raw_df['left_bound_x'][1080:-2840]
    trimmed_df['left_bound_y'] = raw_df['left_bound_y'][1080:-2840]
    trimmed_df['left_bound_z'] = raw_df['left_bound_z'][1080:-2840]
    trimmed_df['right_bound_x'] = raw_df['right_bound_x'][2870:-550]
    trimmed_df['right_bound_y'] = raw_df['right_bound_y'][2870:-550]
    trimmed_df['right_bound_z'] = raw_df['right_bound_z'][2870:-550]
    # Start creation of unique track bounds dataframe
    unique_df = trimmed_df.drop_duplicates(keep='first')
    left_df = get_evenly_spaced_points_on_spline(unique_df, ['left_bound_x','left_bound_y','left_bound_z'], 1000)
    right_df = get_evenly_spaced_points_on_spline(unique_df, ['right_bound_x','right_bound_y','right_bound_z'], 1000)
    merged_df = pd.concat([left_df, right_df], axis=1)
    # End creation of unique track bounds dataframe

    # Start creation of 2d center line
    center_line_df = get_center_line_from_bounds(raw_df)
    # End creation of 2d center line
    # plot_track(raw_df, "raw")
    # plot_track(center_line_df, "center")
    plot_track(merged_df, "unique")
    plt.show()
    # print(merged_df.head())
    trimmed_file = source_dir / 'thunderhill_east_trimmed.csv'
    unique_file = source_dir / 'thunderhill_east_unique.csv'
    center_line_file = source_dir / 'thunderhill_east_2d.csv'
    trimmed_df.to_csv(str(trimmed_file), index=False)
    merged_df.to_csv(str(unique_file), index=False)
    center_line_df.to_csv(str(center_line_file), index=False)

if __name__ == '__main__':
    main()