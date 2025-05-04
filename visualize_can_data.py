import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def plot_can_features(data_filepath, output_dir='output_plots', num_timesteps=500):
    """
    Generates plots for features in a CAN dataset CSV file, mimicking the example style.

    Args:
        data_filepath (str): Path to the input CSV data file.
        output_dir (str): Directory to save the plot.
        num_timesteps (int): Number of initial timesteps to plot.
    """
    print(f"Processing data file: {data_filepath}")

    if not os.path.exists(data_filepath):
        print(f"Error: Data file not found at {data_filepath}")
        return

    # Load data using pandas - assuming comma-separated, potentially with header
    try:
        print("Loading data...")
        # Try reading with header=0 first
        df = pd.read_csv(data_filepath, header=0, nrows=num_timesteps)

        # Attempt to convert all columns to numeric, coercing errors
        print("Converting columns to numeric...")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Optional: Drop columns that are entirely NaN after conversion?
        # df.dropna(axis=1, how='all', inplace=True)

        # Use only the requested number of timesteps if file is longer
        if len(df) > num_timesteps:
            df = df.iloc[:num_timesteps]
        elif len(df) < num_timesteps:
             print(f"Warning: File has only {len(df)} rows, plotting all available.")
             num_timesteps = len(df)

        if df.empty:
             print("Error: Data file is empty or contains no numeric data.")
             return

    except Exception as e:
        # If header=0 fails, maybe try header=None again?
        # Or just report the error
        print(f"Error loading or processing data file {data_filepath}: {e}")
        return

    num_features = df.shape[1]
    print(f"Found {num_features} numeric features. Plotting first {num_timesteps} timesteps...")

    # Determine layout (2 columns like example)
    n_cols = 2
    n_rows = (num_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 3), squeeze=False) # Adjust figsize as needed
    axes = axes.flatten()

    time_steps = np.arange(len(df))

    for i in range(num_features):
        col_name = df.columns[i]
        ax = axes[i]
        # Use column name for data access now
        feature_data = df[col_name].dropna() # Drop NaNs for unique/min/max checks

        if feature_data.empty:
             print(f"Skipping empty or fully NaN feature: {col_name}")
             ax.set_title(f"Feature {col_name} (Empty)")
             ax.text(0.5, 0.5, 'No Numeric Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             continue # Skip plotting for this feature

        # Heuristic to determine if feature is binary after numeric conversion & dropna
        unique_vals = feature_data.unique()
        is_binary = len(unique_vals) <= 2 and feature_data.min() >= 0 and feature_data.max() <= 1

        if is_binary:
            # Use original column name if available and sensible, else generic
            title = f"Binary Feature \'{col_name}\'" if isinstance(col_name, str) else f"Binary Feature {i+1}"
            ax.set_title(title)
            ax.set_ylabel("Event Occurrence")
        else:
            title = f"Continuous Feature \'{col_name}\'" if isinstance(col_name, str) else f"Continuous Feature {i+1}"
            ax.set_title(title)
            ax.set_ylabel("Measurement Values")

        # Plot original data from the column (including potential NaNs, plot handles them)
        ax.plot(time_steps, df[col_name], color='red')
        ax.set_xlabel("Time")
        ax.grid(True)

        # Potentially set Y limits for binary features
        if is_binary:
             ax.set_ylim(-0.1, 1.1)

    # Hide unused subplots
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout slightly

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot as PDF for high quality
    base_filename = os.path.splitext(os.path.basename(data_filepath))[0]
    plot_filename = f"{base_filename}_features_{num_timesteps}steps.pdf"
    plot_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig) # Close the figure to free memory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize CAN Bus Dataset Features')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the input CSV data file.')
    parser.add_argument('--output_dir', type=str, default='output_plots', help='Directory to save the plot.')
    parser.add_argument('--timesteps', type=int, default=500, help='Number of initial timesteps to plot.')

    args = parser.parse_args()

    plot_can_features(args.data_file, args.output_dir, args.timesteps) 