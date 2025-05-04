import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import re
import csv # For reading the anomaly CSV
import json # For parsing the anomaly sequences string
from matplotlib.patches import Patch # For legend

# Apply a cleaner plot style
# plt.style.use('seaborn-v0_8-whitegrid')

def parse_interpretation_details(filepath):
    """Parses interpretation label file returning spans and affected features.
    Returns: List of tuples: [(start_idx, end_idx, [affected_feature_indices]), ...]
             Indices are 0-based.
    """
    anomaly_details = []
    if not os.path.exists(filepath):
        print(f"Warning: Interpretation label file not found at {filepath}.")
        return anomaly_details
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                # Use regex to extract range and feature list
                match = re.match(r'(\d+)-(\d+):(.*)', line)
                if match:
                    start_str, end_str, features_str = match.groups()
                    start, end = int(start_str), int(end_str)
                    affected_indices = []
                    if features_str.strip(): # Check if feature list is not empty
                        try:
                            # Features are 1-based in file
                            affected_indices = [int(f.strip()) - 1 for f in features_str.split(',')]
                        except ValueError:
                            print(f"Warning: Could not parse feature list '{features_str}' in line: {line}")

                    # Assume interpretation labels are 1-based and inclusive
                    # Adjust to 0-based indices for start, keep end as is for slicing logic later
                    start_idx = max(0, start - 1)
                    # Store end as the *inclusive* end from the file for range checks
                    end_idx_inclusive = end - 1

                    # Store tuple: (0-based start, 0-based INCLUSIVE end, list of 0-based affected features)
                    anomaly_details.append((start_idx, end_idx_inclusive, affected_indices))
                else:
                    print(f"Warning: Could not parse line in interpretation file: {line}")
    except Exception as e:
        print(f"Error reading or parsing interpretation label file {filepath}: {e}")
    return anomaly_details

# Function to load data (handles .npy and potentially others)
def load_data(filepath):
    """Loads time series data, trying .npy first."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    if filepath.endswith('.npy'):
        print(f"Loading NumPy data from: {filepath}")
        try:
            data = np.load(filepath)
            # Ensure data is 2D (timesteps, features)
            if data.ndim == 1:
                data = data.reshape(-1, 1) # Reshape 1D array to 2D with one feature
            elif data.ndim != 2:
                raise ValueError(f"NumPy array at {filepath} has unexpected dimensions: {data.ndim}")
            return data
        except Exception as e:
            raise IOError(f"Error loading NumPy file {filepath}: {e}")
    elif filepath.endswith('.csv') or filepath.endswith('.txt'):
        print(f"Loading CSV/TXT data from: {filepath}")
        try:
            # Use pandas for robustness with CSV/TXT
            df = pd.read_csv(filepath, header=None)
            return df.values # Return as numpy array
        except Exception as e:
             raise IOError(f"Error loading CSV/TXT file {filepath}: {e}")
    else:
        raise ValueError(f"Unsupported file format: {filepath}. Only .npy, .csv, .txt are supported.")

# Modified label parsing function for labeled_anomalies.csv
def parse_anomaly_csv(label_csv_path, target_chan_id, data_length):
    """Parses labeled_anomalies.csv to create a binary label array for a specific channel."""
    labels = np.zeros(data_length, dtype=int)
    if not os.path.exists(label_csv_path):
        print(f"Warning: Anomaly CSV file not found at {label_csv_path}. Assuming no anomalies.")
        return labels

    found_channel = False
    try:
        with open(label_csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['chan_id'] == target_chan_id:
                    found_channel = True
                    anomaly_sequences_str = row['anomaly_sequences']
                    try:
                        # The string looks like a Python list literal, try json.loads
                        # Need to replace single quotes with double quotes for valid JSON
                        anomaly_sequences_str = anomaly_sequences_str.replace("'", '"')
                        anomaly_sequences = json.loads(anomaly_sequences_str)

                        for seq in anomaly_sequences:
                            if len(seq) == 2:
                                start, end = seq
                                # Assume indices are 1-based and inclusive from CSV
                                # Adjust to 0-based index, exclusive end for slicing
                                start_idx = max(0, start - 1)
                                end_idx = min(data_length, end) # end index from CSV is inclusive, so use as is for exclusive end
                                if start_idx < end_idx:
                                    labels[start_idx:end_idx] = 1
                            else:
                                print(f"Warning: Invalid sequence format in {label_csv_path} for {target_chan_id}: {seq}")
                    except json.JSONDecodeError as json_err:
                         print(f"Warning: Could not parse anomaly sequence string for {target_chan_id}: '{row['anomaly_sequences']}'. Error: {json_err}")
                    except Exception as parse_err:
                        print(f"Warning: Error processing anomaly sequence for {target_chan_id}: {parse_err}")
                    break # Stop after finding the target channel

        if not found_channel:
            print(f"Warning: Channel ID '{target_chan_id}' not found in {label_csv_path}. Assuming no anomalies.")

    except Exception as e:
        print(f"Error reading or parsing anomaly CSV file {label_csv_path}: {e}")

    return labels

def plot_dataset_features(dataset_name, machine_id, n_timesteps=100, start_timestep=0, data_dir='data', output_dir='output_plots', features_str=None):
    """
    Generates plots comparing train and test data features, shading only affected features during anomalies.
    """
    print(f"Processing Dataset: {dataset_name}, Machine/Channel: {machine_id}, Timesteps: {start_timestep} to {start_timestep + n_timesteps}")

    # Determine file extension (try npy first, then csv/txt)
    # Note: This assumes train/test files have the SAME name+extension
    # Construct potential filenames based on common patterns
    potential_train_npy = os.path.join(data_dir, dataset_name, 'train', f"machine-{machine_id}.npy") # SMD pattern
    potential_train_csv = os.path.join(data_dir, dataset_name, 'train', f"machine-{machine_id}.csv")
    potential_train_txt = os.path.join(data_dir, dataset_name, 'train', f"machine-{machine_id}.txt")
    potential_smap_npy = os.path.join(data_dir, dataset_name, 'train', f"{machine_id}.npy") # SMAP pattern

    file_pattern = None
    if os.path.exists(potential_train_npy):
        file_ext = '.npy'
        file_pattern = f"machine-{machine_id}"
    elif os.path.exists(potential_train_csv):
        file_ext = '.csv'
        file_pattern = f"machine-{machine_id}"
    elif os.path.exists(potential_train_txt):
        file_ext = '.txt'
        file_pattern = f"machine-{machine_id}"
    elif os.path.exists(potential_smap_npy):
        file_ext = '.npy'
        file_pattern = f"{machine_id}" # Use SMAP pattern
    # Add elif checks for potential_smap_csv, potential_smap_txt if needed
    else:
         print(f"Error: Could not find train data file for {machine_id} with known patterns (.npy, .csv, .txt) in {os.path.join(data_dir, dataset_name, 'train')}")
         return

    # Construct file paths using determined pattern and extension
    base_path = os.path.join(data_dir, dataset_name)
    machine_file_base = file_pattern # Use the detected pattern
    train_path = os.path.join(base_path, 'train', f"{machine_file_base}{file_ext}") # Re-enable train_path
    test_path = os.path.join(base_path, 'test', f"{machine_file_base}{file_ext}")

    # Determine label file path
    anomaly_csv_path = os.path.join(base_path, 'labeled_anomalies.csv')
    # Use the determined base name for interpretation label path
    interpretation_label_path = os.path.join(base_path, 'interpretation_label', f"{machine_file_base}.txt")

    use_anomaly_csv = os.path.exists(anomaly_csv_path) and dataset_name in ['SMAP_MSL'] # Be explicit for now
    use_interpretation = os.path.exists(interpretation_label_path)

    # Check if data files exist
    if not os.path.exists(test_path):
        print(f"Error: Test file not found at {test_path}")
        return
    # Add check for train path existence
    if not os.path.exists(train_path):
        print(f"Error: Train file not found at {train_path}")
        return

    # Load data
    try:
        train_data = load_data(train_path) # Re-enable loading train data
        test_data = load_data(test_path)

        anomaly_details = [] # List to hold [(start, end, [features]), ...]
        labels = np.zeros(len(test_data), dtype=int) # Initialize labels array

        if use_anomaly_csv:
            print(f"Using anomaly CSV: {anomaly_csv_path}")
            labels = parse_anomaly_csv(anomaly_csv_path, machine_id, len(test_data))
        elif use_interpretation:
            print(f"Using interpretation label file: {interpretation_label_path}")
            anomaly_details = parse_interpretation_details(interpretation_label_path)
            # Populate the binary labels array from anomaly_details
            print("Populating binary labels array from details...")
            for start_idx, end_idx_inclusive, _ in anomaly_details:
                # Ensure indices are within bounds of the test data length
                start = max(0, start_idx)
                # Convert inclusive end to exclusive end for slicing
                end = min(len(test_data), end_idx_inclusive + 1)
                if start < end:
                    labels[start:end] = 1
            print(f"Generated labels array with {np.sum(labels)} anomaly points.")
        else:
            print("Error: No label file found.")
            return

    except (FileNotFoundError, IOError, ValueError) as e:
        print(f"Error loading data or labels: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data/label loading: {e}")
        return

    # Ensure data is not empty
    if train_data.size == 0 or test_data.size == 0:
        print("Error: One or more data files are empty or failed to load.")
        return

    # Get *original* number of features before potential filtering
    original_n_features = test_data.shape[1] if test_data.ndim == 2 else 1
    # Add warning if train features don't match test features
    if train_data.ndim == 2 and train_data.shape[1] != original_n_features:
        print(f"Warning: Train ({train_data.shape[1]}) and Test ({original_n_features}) datasets have different number of features before filtering.")
        # Simple handling: use min features. More complex logic might be needed.
        # This might not be necessary if filtering happens correctly
        # original_n_features = min(train_data.shape[1], original_n_features)

    # --- Feature Selection --- START
    # Revert feature selection - plot all features for now
    all_feature_indices = list(range(original_n_features))
    selected_feature_indices = all_feature_indices
    n_features = original_n_features
    # --- Feature Selection --- END

    # Filter data based on selected features (No filtering applied now)
    # train_data = train_data[:, selected_feature_indices]
    # test_data = test_data[:, selected_feature_indices]

    # --- Data Slicing --- START
    end_timestep = start_timestep + n_timesteps

    # Slice selected feature data
    train_plot_data = train_data[start_timestep:end_timestep, :] # Already filtered by columns
    test_plot_data = test_data[start_timestep:end_timestep, :]   # Already filtered by columns
    labels_plot = labels[start_timestep:end_timestep]

    # Check slice validity
    if test_plot_data.size == 0:
        print(f"Error: No test data available in the specified window {start_timestep}-{end_timestep}. Max index is {len(test_data)-1}")
        return
    # Add check for train data slice emptiness
    if train_plot_data.size == 0:
        # This warning might be less relevant now, or check len(train_data)
        print(f"Warning: Not enough initial train data ({len(train_data)}) to compare with the window length ({len(test_plot_data)})")

    # --- Data Slicing --- END

    # Create the plot
    # Absolute time steps for the x-axis of the test window
    time_steps_absolute = np.arange(start_timestep, start_timestep + len(test_plot_data))
    anomaly_indices_relative = np.where(labels_plot == 1)[0]
    anomaly_time_steps_absolute = time_steps_absolute[anomaly_indices_relative]

    # Ensure data is 2D
    test_plot_data_2d = test_plot_data if test_plot_data.ndim == 2 else test_plot_data.reshape(-1, 1)
    train_plot_data_2d = train_plot_data if train_plot_data.ndim == 2 else train_plot_data.reshape(-1, 1)
    # We don't need anomaly_data_points if using background shading

    # Use the filtered number of features
    num_features_to_plot = n_features

    # Subplot layout - Adjust based on potentially fewer features
    n_cols = 5 # Keep 5 columns unless only 1 feature
    if num_features_to_plot == 1:
        n_cols = 1
    elif num_features_to_plot <= n_cols:
         n_cols = num_features_to_plot # Use fewer columns if fewer features than default

    n_rows = (num_features_to_plot + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.5), sharex=True, squeeze=False)
    axes = axes.flatten()

    # --- Anomaly Shading --- START
    # Find contiguous anomalous blocks within the window
    anomaly_spans = []
    if len(anomaly_indices_relative) > 0:
        start_block = anomaly_time_steps_absolute[0]
        end_block = start_block
        for i in range(1, len(anomaly_indices_relative)):
            current_time = anomaly_time_steps_absolute[i]
            # Check if the current step continues the block (consecutive time steps)
            if current_time == end_block + 1:
                end_block = current_time # Extend the block
            else:
                # End previous block and start a new one
                anomaly_spans.append((start_block, end_block))
                start_block = current_time
                end_block = start_block
        # Add the last block
        anomaly_spans.append((start_block, end_block))
    # --- Anomaly Shading --- END

    print(f"Plotting {num_features_to_plot} features...")
    # Update print statement regarding anomalies
    print(f"Highlighting {len(anomaly_spans)} anomalous period(s) within window {start_timestep}-{start_timestep + len(test_plot_data) - 1}.")

    num_shaded_features = 0 # Counter for logging

    for i in range(num_features_to_plot):
        original_feature_index = selected_feature_indices[i]
        ax = axes[i]
        # Plot initial train data segment
        if i < train_plot_data_2d.shape[1] and train_plot_data_2d.size > 0:
             # Simplify legend label
             train_label = 'Train Data' if i == 0 else ""
             ax.plot(time_steps_absolute[:len(train_plot_data_2d)], train_plot_data_2d[:, i], color='green', label=train_label, alpha=1.0)

        # Plot test data for the window
        # Simplify legend label
        test_label = 'Test Data' if i == 0 else ""
        ax.plot(time_steps_absolute, test_plot_data_2d[:, i], color='blue', label=test_label, alpha=1.0)

        # --- Apply Shading Conditionally --- START
        feature_is_anomalous_in_window = False
        for anom_start_idx, anom_end_idx_inclusive, affected_features in anomaly_details:
            # Check if the CURRENT feature is affected by THIS anomaly span
            if original_feature_index in affected_features:
                # Determine overlap of this anomaly span with the plotting window
                window_start = start_timestep
                window_end = start_timestep + len(test_plot_data_2d) # Exclusive end
                
                overlap_start = max(anom_start_idx, window_start)
                # Convert inclusive end to exclusive end for comparison/axvspan
                overlap_end = min(anom_end_idx_inclusive + 1, window_end)

                # If there is an overlap, apply shading
                if overlap_start < overlap_end:
                    ax.axvspan(overlap_start, overlap_end, color='red', alpha=0.2, zorder=0)
                    feature_is_anomalous_in_window = True
        # --- Apply Shading Conditionally --- END
        
        if feature_is_anomalous_in_window:
             num_shaded_features += 1

        # Use original feature index + 1 for title
        ax.set_title(f'Feature {original_feature_index + 1}')
        ax.set_xlabel("Time")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)

    # Legend Placement
    # Need to manually add an entry for the shaded region if desired
    handles, labels_legend = [], []
    if num_features_to_plot > 0:
        temp_handles, temp_labels = axes[0].get_legend_handles_labels()
        handles.extend(temp_handles)
        labels_legend.extend(temp_labels)
        if anomaly_spans:
            # Check if 'Anomaly Period' label already exists from a previous run/plot element
            if 'Anomaly Period' not in labels_legend:
                anomaly_patch = Patch(color='red', alpha=0.2, label='Anomaly Period')
                # Check if the handle list is empty before appending (shouldn't be if features plotted)
                if handles:
                    handles.append(anomaly_patch)
                    labels_legend.append('Anomaly Period')
                else: # Handle edge case where no lines were plotted but anomalies exist
                    handles = [anomaly_patch]
                    labels_legend = ['Anomaly Period']

    legend_dict = dict(zip(labels_legend, handles))

    legend_ax_found = False
    for j in range(num_features_to_plot, len(axes)):
        # Place legend if dict is not empty
        if not legend_ax_found and legend_dict:
            axes[j].legend(legend_dict.values(), legend_dict.keys(), loc='center')
            axes[j].axis('off')
            legend_ax_found = True
        else:
             # Keep hiding unused axes
             fig.delaxes(axes[j])

    # Fallback legend (unlikely needed now)
    if not legend_ax_found and legend_dict:
         fig.legend(legend_dict.values(), legend_dict.keys(), loc='upper right')

    # Improve layout
    # Removed suptitle
    plt.tight_layout() # Use default tight_layout without rect adjustment initially

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot as PDF
    # Update filename extension
    plot_filename = f"{dataset_name}_channel_{machine_id}_features_{start_timestep}_to_{start_timestep + len(test_plot_data) - 1}.pdf"
    plot_path = os.path.join(output_dir, plot_filename)
    try:
        # Save as PDF instead of PNG
        plt.savefig(plot_path, format='pdf', bbox_inches='tight') # Use bbox_inches='tight' for PDF
        print(f"Plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig)

    print(f"Applied anomaly shading to {num_shaded_features} feature subplots.")

if __name__ == "__main__":
    # Updated description
    parser = argparse.ArgumentParser(description='Visualize a Window of Dataset Features (Supports NPY/CSV/TXT & different label formats)')
    # Change 'machine' help text to be more general
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset directory (e.g., SMD, SMAP_MSL)')
    parser.add_argument('--machine', type=str, required=True, help='Machine or Channel ID (e.g., 1-1, D-14)')
    # ... (keep other arguments: timesteps, start_timestep, data_dir, output_dir) ...
    parser.add_argument('--timesteps', type=int, default=100, help='Number of timesteps in the window to plot')
    parser.add_argument('--start_timestep', type=int, default=0, help='Starting timestep for the window')
    parser.add_argument('--data_dir', type=str, default='data', help='Root directory for datasets')
    parser.add_argument('--output_dir', type=str, default='output_plots', help='Directory to save plots')
    parser.add_argument('--features', type=str, default=None, help='Comma-separated list of 1-based feature indices to plot (e.g., \'1,5,10\')')

    args = parser.parse_args()
    plot_dataset_features(args.dataset, args.machine, args.timesteps, args.start_timestep, args.data_dir, args.output_dir, features_str=None)

    # Add more calls here for other datasets or machines if needed
    # Example: plot_dataset_features('SMD', '1-2', args.timesteps, args.data_dir, args.output_dir) 