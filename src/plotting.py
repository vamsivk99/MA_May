import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np

# plt.style.use(['science', 'ieee']) # Temporarily comment out for testing
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs('plots', exist_ok=True)

def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotter(name, y_true, y_pred, ascore, labels):
	if 'TranAD' in name: y_true = torch.roll(y_true, 1, 0)
	os.makedirs(os.path.join('plots', name), exist_ok=True)
	pdf = PdfPages(f'plots/{name}/output.pdf')
	
	# Ensure all inputs are 2D arrays
	if len(y_true.shape) == 1: y_true = y_true.reshape(-1, 1)
	if len(y_pred.shape) == 1: y_pred = y_pred.reshape(-1, 1)
	if len(ascore.shape) == 1: ascore = ascore.reshape(-1, 1)
	if len(labels.shape) == 1: labels = labels.reshape(-1, 1)
	
	# If any input has only one dimension, repeat it for all features
	n_features = max(y_true.shape[1], y_pred.shape[1], labels.shape[1])
	if y_true.shape[1] == 1: y_true = np.repeat(y_true, n_features, axis=1)
	if y_pred.shape[1] == 1: y_pred = np.repeat(y_pred, n_features, axis=1)
	if ascore.shape[1] == 1: ascore = np.repeat(ascore, n_features, axis=1)
	if labels.shape[1] == 1: labels = np.repeat(labels, n_features, axis=1)
	
	for dim in range(n_features):
		y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim]
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
		ax1.set_ylabel('Value')
		ax1.set_title(f'Dimension = {dim}')
		# if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
		ax1.plot(smooth(y_t), linewidth=0.2, label='True')
		ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
		ax3 = ax1.twinx()
		ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
		ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
		if dim == 0: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
		ax2.plot(smooth(a_s), linewidth=0.2, color='g')
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel('Anomaly Score')
		pdf.savefig(fig)
		plt.close()
	pdf.close()
