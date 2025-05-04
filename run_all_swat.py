import subprocess
import pandas as pd
import os
import sys
import time
from datetime import datetime

# List of models to run
models = [
    'LSTMAutoencoder',
    'GRUAutoencoder',
    'ConvLSTMAutoencoder',
    'ConvGRUAutoencoder',
    'ConvLSTMAttentionAutoencoder',
    'ConvGRUAttentionAutoencoder',
    'TransformerEncoderDecoder',
    'VanillaTransformer',
    'TranAD'
]

# Window sizes for different model types
window_sizes = {
    'LSTMAutoencoder': 100,
    'GRUAutoencoder': 100,
    'ConvLSTMAutoencoder': 100,
    'ConvGRUAutoencoder': 100,
    'ConvLSTMAttentionAutoencoder': 100,
    'ConvGRUAttentionAutoencoder': 100,
    'TransformerEncoderDecoder': 200,  # Transformers can handle longer sequences
    'VanillaTransformer': 200,
    'TranAD': 200
}

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Get Python executable path
python_exe = sys.executable

# Create a results log file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f'results/swat_results_{timestamp}.log'

# Run each model
for model in models:
    print(f"\n=== Running {model} on SWaT with window size {window_sizes[model]} ===")
    start_time = time.time()
    
    # Construct the command with window size parameter
    cmd = f'"{python_exe}" main.py --model {model} --dataset SWaT --window {window_sizes[model]}'
    
    try:
        # Run the command and capture output
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Write results to log file
        with open(results_file, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Window Size: {window_sizes[model]}\n")
            f.write(f"Runtime: {runtime:.2f} seconds\n")
            f.write(f"Command: {cmd}\n")
            f.write(f"Output:\n{result.stdout}\n")
            if result.stderr:
                f.write(f"Errors:\n{result.stderr}\n")
        
        print(f"Completed {model} in {runtime:.2f} seconds")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running {model}: {e}")
        with open(results_file, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Window Size: {window_sizes[model]}\n")
            f.write(f"Error: {str(e)}\n")
        continue

print(f"\nAll runs completed! Results saved to {results_file}") 