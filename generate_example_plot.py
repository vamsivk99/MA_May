import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def generate_synthetic_data(num_timesteps=500):
    """Generates synthetic data mimicking the example plot."""
    time = np.arange(num_timesteps)
    data = {}

    # Continuous Feature 1: Dip, spike, decay, smaller spike
    f1 = np.ones(num_timesteps) * 1100
    f1[0:20] = np.linspace(1075, 1050, 20)
    f1[20:75] = np.linspace(1050, 1225, 55)
    f1[75:300] = np.linspace(1225, 1125, 225)
    f1[300:320] = np.linspace(1125, 1075, 20)
    f1[320:340] = np.linspace(1075, 1125, 20)
    f1[340:] = np.linspace(1125, 1080, num_timesteps - 340)
    f1 = np.clip(f1, 1050, 1225)
    data['cont1'] = f1

    # Continuous Feature 2: Similar to 1, slightly different timings/values
    f2 = np.ones(num_timesteps) * 1100
    f2[0:25] = np.linspace(1080, 1055, 25)
    f2[25:80] = np.linspace(1055, 1220, 55) # Slightly lower peak
    f2[80:310] = np.linspace(1220, 1120, 230)
    f2[310:330] = np.linspace(1120, 1060, 20)
    f2[330:350] = np.linspace(1060, 1115, 20)
    f2[350:] = np.linspace(1115, 1070, num_timesteps - 350)
    f2 = np.clip(f2, 1050, 1225)
    data['cont2'] = f2

    # Continuous Feature 3: Noisy, high frequency
    f3_base = np.linspace(830, 840, num_timesteps) + np.sin(time / 10) * 15
    f3 = f3_base
    f3 = np.clip(f3, 790, 870)
    data['cont3'] = f3

    # Continuous Feature 4: Oscillatory, increasing amplitude
    f4 = np.sin(time / 30) * (20 + time / 15) + 40
    f4[0:50] = np.linspace(0, 25, 50)
    f4 = np.clip(f4, 0, 90)
    data['cont4'] = f4

    # Binary Feature 1: Pulses
    b1 = np.zeros(num_timesteps)
    b1[20:80] = 1.0
    b1[310:325] = 1.0
    data['bin1'] = b1

    # Binary Feature 2: Similar pulses
    b2 = np.zeros(num_timesteps)
    b2[25:75] = 1.0 # Slightly shifted
    b2[315:335] = 1.0 # Slightly shifted/longer
    data['bin2'] = b2

    return data, time

def plot_synthetic_can(output_dir='output_plots', num_timesteps=500):
    """Generates a plot from synthetic data mimicking the example style."""

    print("Generating synthetic data...")
    synthetic_data, time_steps = generate_synthetic_data(num_timesteps)

    print("Plotting synthetic data...")
    # Fixed layout: 3 rows, 2 columns
    n_rows = 3
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 3), squeeze=False)
    axes = axes.flatten()

    plot_map = {
        0: ('cont1', "Continuous Feature 1", "Measurement Values"),
        1: ('cont2', "Continuous Feature 2", "Measurement Values"),
        2: ('cont3', "Continuous Feature 3", "Measurement Values"),
        3: ('cont4', "Continuous Feature 4", "Measurement Values"),
        4: ('bin1', "Binary Feature 1", "Event Occurrence"),
        5: ('bin2', "Binary Feature 2", "Event Occurrence")
    }

    for i in range(len(plot_map)):
        ax = axes[i]
        key, title, ylabel = plot_map[i]
        feature_data = synthetic_data[key]

        ax.plot(time_steps, feature_data, color='red')
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.grid(True)

        if key.startswith('bin'):
             ax.set_ylim(-0.1, 1.1) # Set Y limits for binary features

    # Hide unused subplots (none in this fixed 3x2 layout)
    # for j in range(len(plot_map), len(axes)):
    #     fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout slightly

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot as PDF for high quality
    plot_filename = "synthetic_can_features_example.pdf"
    plot_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        print(f"Synthetic plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig) # Close the figure to free memory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Synthetic CAN Bus Plot Example')
    parser.add_argument('--output_dir', type=str, default='output_plots', help='Directory to save the plot.')
    parser.add_argument('--timesteps', type=int, default=500, help='Number of timesteps for the synthetic data.')

    args = parser.parse_args()

    plot_synthetic_can(args.output_dir, args.timesteps) 