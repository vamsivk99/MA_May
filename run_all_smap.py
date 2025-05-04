import subprocess
import pandas as pd
import os
import sys

# Read SMAP entities
df = pd.read_csv('data/SMAP_MSL/labeled_anomalies.csv')
smap_entities = df[df['spacecraft'] == 'SMAP']['chan_id'].unique()

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

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Get Python executable path
python_exe = sys.executable

# Run each model on each entity
for model in models:
    print(f"\n=== Running {model} ===")
    for entity in smap_entities:
        print(f"\n--- Running on {entity} ---")
        cmd = f'"{python_exe}" main.py --model {model} --dataset SMAP --entity {entity}'
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {model} on {entity}: {e}")
            continue

print("\nAll runs completed!") 