import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.metrics import mean_squared_error # Example error metric

def load_data_generic(filepath):
    """Loads time series data from .npy, .csv, or .txt"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"Loading data from: {filepath}")
    if filepath.endswith('.npy'):
        try:
            data = np.load(filepath)
        except Exception as e:
            raise IOError(f"Error loading NumPy file {filepath}: {e}")
    elif filepath.endswith('.csv') or filepath.endswith('.txt'):
        try:
            # Try with header first, then without if it fails
            try:
                df = pd.read_csv(filepath, header=0)
            except Exception:
                print(f"Could not read {filepath} with header, trying without header.")
                df = pd.read_csv(filepath, header=None)
            
            # Convert numeric columns, coerce others to NaN
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # Drop fully NaN columns (like potential ID columns)
            df.dropna(axis=1, how='all', inplace=True)
            data = df.values
        except Exception as e:
             raise IOError(f"Error loading CSV/TXT file {filepath}: {e}")
    else:
        raise ValueError(f"Unsupported file format: {filepath}. Only .npy, .csv, .txt are supported.")

    # Ensure data is 2D numpy array
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim != 2:
         raise ValueError(f"Loaded data from {filepath} has unexpected dimensions: {data.ndim}")
    
    print(f"Loaded data shape: {data.shape}")
    return data


def plot_reconstruction(original_data_path, reconstructed_data_path, output_dir='output_plots', 
                        timesteps_to_plot=None, start_timestep=0):
    """
    Generates plots comparing original and reconstructed signals.

    Args:
        original_data_path (str): Path to the original data file.
        reconstructed_data_path (str): Path to the reconstructed data file.
        output_dir (str): Directory to save the plot.
        timesteps_to_plot (int, optional): Number of timesteps to plot from the start. Plots all if None.
        start_timestep (int): Starting timestep for the plot window (used if timesteps_to_plot is set).
    """
    try:
        original_data = load_data_generic(original_data_path)
        reconstructed_data = load_data_generic(reconstructed_data_path)
    except (FileNotFoundError, IOError, ValueError) as e:
        print(f"Error: {e}")
        return

    # --- Data Alignment and Slicing ---
    min_len = min(len(original_data), len(reconstructed_data))
    if len(original_data) != len(reconstructed_data):
        print(f"Warning: Original ({len(original_data)}) and reconstructed ({len(reconstructed_data)}) data have different lengths. Truncating to minimum length: {min_len}")
        original_data = original_data[:min_len]
        reconstructed_data = reconstructed_data[:min_len]

    num_features = original_data.shape[1]
    if reconstructed_data.shape[1] != num_features:
         print(f"Error: Original ({num_features}) and reconstructed ({reconstructed_data.shape[1]}) data have different number of features.")
         return

    # Apply windowing/slicing if requested
    plot_label = "Full"
    if timesteps_to_plot is not None:
        end_timestep = start_timestep + timesteps_to_plot
        if start_timestep >= min_len:
             print(f"Error: Start timestep {start_timestep} is beyond data length {min_len}.")
             return
        end_timestep = min(end_timestep, min_len) # Adjust end if it exceeds data length
        
        original_data = original_data[start_timestep:end_timestep]
        reconstructed_data = reconstructed_data[start_timestep:end_timestep]
        time_steps = np.arange(start_timestep, end_timestep)
        plot_label = f"{start_timestep}_to_{end_timestep-1}"
        print(f"Plotting window: Timesteps {start_timestep} to {end_timestep-1}")
    else:
         time_steps = np.arange(min_len)
         print(f"Plotting full data ({min_len} timesteps)")

    if original_data.size == 0:
         print("Error: No data available in the specified window.")
         return
         
    # --- Plotting ---
    print("Plotting features...")
    n_cols = 5 # As per example image
    n_rows = (num_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2.5), sharex=True, squeeze=False)
    axes = axes.flatten()

    for i in range(num_features):
        ax = axes[i]
        
        # Plot original and reconstructed signals
        ax.plot(time_steps, original_data[:, i], color='blue', label='Original Signal' if i == 0 else "", alpha=0.8, linewidth=1.0)
        ax.plot(time_steps, reconstructed_data[:, i], color='red', label='Reconstructed Signal' if i == 0 else "", alpha=0.8, linewidth=1.0)
        
        ax.set_title(f'Feature {i+1}')
        ax.set_xlabel("Time")
        # Maybe remove y-label for cleaner look like example?
        # ax.set_ylabel("Value") 
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='x', rotation=45)

    # Hide unused subplots
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    # Create legend - place it in an empty subplot if possible
    handles, labels = [], []
    if num_features > 0:
        handles, labels = axes[0].get_legend_handles_labels()

    legend_ax_found = False
    if handles: # Only create legend if there's something to show
        for j in range(num_features, len(axes)):
            if not legend_ax_found:
                axes[j].legend(handles, labels, loc='center')
                axes[j].axis('off')
                legend_ax_found = True
            else:
                # Keep hiding other unused axes
                 fig.delaxes(axes[j])
        # Fallback legend if no empty space
        if not legend_ax_found:
             fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1.0))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # --- Saving ---
    os.makedirs(output_dir, exist_ok=True)
    # Extract model name from reconstructed file path if possible for filename
    recon_filename_base = os.path.splitext(os.path.basename(reconstructed_data_path))[0]
    plot_filename = f"{recon_filename_base}_comparison_{plot_label}.pdf"
    plot_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        print(f"Reconstruction comparison plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Original vs. Reconstructed Signals')
    parser.add_argument('--original', type=str, required=True, help='Path to the original data file (.npy, .csv, .txt).')
    parser.add_argument('--reconstructed', type=str, required=True, help='Path to the reconstructed data file (.npy, .csv, .txt).')
    parser.add_argument('--output_dir', type=str, default='output_plots', help='Directory to save the plot.')
    parser.add_argument('--timesteps', type=int, default=None, help='Number of timesteps to plot (plots all if not specified).')
    parser.add_argument('--start', type=int, default=0, help='Start timestep for plotting window (used with --timesteps).')

    args = parser.parse_args()

    plot_reconstruction(
        original_data_path=args.original,
        reconstructed_data_path=args.reconstructed,
        output_dir=args.output_dir,
        timesteps_to_plot=args.timesteps,
        start_timestep=args.start
    ) 