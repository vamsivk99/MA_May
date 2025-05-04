import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
# import dgl # Removed
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *
from src.constants import *
import math # Added earlier
torch.manual_seed(1)

# Attempt to import dgl, but don't fail if it's not installed or configured correctly
# try:
#     import dgl
#     DGL_AVAILABLE = True
# except ImportError:
#     print("Warning: DGL not found or could not be imported. Models requiring DGL (e.g., GDN, MTAD_GAT) will not work.")
#     DGL_AVAILABLE = False
# except FileNotFoundError as e:
#     print(f"Warning: DGL installed but graphbolt library issue: {e}. Models requiring DGL will not work.")
#     DGL_AVAILABLE = False

# Define device from main script scope if available, else default to CPU
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except NameError:
    device = torch.device("cpu")

## Separate LSTM for each variable
class LSTM_Univariate(nn.Module):
	def __init__(self, feats):
		super(LSTM_Univariate, self).__init__()
		self.name = 'LSTM_Univariate'
		self.lr = 0.002
		self.n_feats = feats
		self.n_hidden = 1
		self.lstm = nn.ModuleList([nn.LSTM(1, self.n_hidden) for i in range(feats)])

	def forward(self, x):
		hidden = [(torch.rand(1, 1, self.n_hidden, dtype=torch.float64), 
			torch.randn(1, 1, self.n_hidden, dtype=torch.float64)) for i in range(self.n_feats)]
		outputs = []
		for i, g in enumerate(x):
			multivariate_output = []
			for j in range(self.n_feats):
				univariate_input = g.view(-1)[j].view(1, 1, -1)
				out, hidden[j] = self.lstm[j](univariate_input, hidden[j])
				multivariate_output.append(2 * out.view(-1))
			output = torch.cat(multivariate_output)
			outputs.append(output)
		return torch.stack(outputs)

## Simple Multi-Head Self-Attention Model
class Attention(nn.Module):
	def __init__(self, feats):
		super(Attention, self).__init__()
		self.name = 'Attention'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_window = 5 # MHA w_size = 5
		self.n = self.n_feats * self.n_window
		self.atts = [ nn.Sequential( nn.Linear(self.n, feats * feats), 
				nn.ReLU(True))	for i in range(1)]
		self.atts = nn.ModuleList(self.atts)

	def forward(self, g):
		for at in self.atts:
			ats = at(g.view(-1)).reshape(self.n_feats, self.n_feats)
			g = torch.matmul(g, ats)		
		return g, ats

## LSTM_AD Model
class LSTM_AD(nn.Module):
	def __init__(self, feats):
		super(LSTM_AD, self).__init__()
		self.name = 'LSTM_AD'
		self.lr = 0.002
		self.n_feats = feats
		self.n_hidden = 64
		self.lstm = nn.LSTM(feats, self.n_hidden)
		self.lstm2 = nn.LSTM(feats, self.n_feats)
		self.fcn = nn.Sequential(nn.Linear(self.n_feats, self.n_feats), nn.Sigmoid())

	def forward(self, x):
		hidden = (torch.rand(1, 1, self.n_hidden, dtype=torch.float64), torch.randn(1, 1, self.n_hidden, dtype=torch.float64))
		hidden2 = (torch.rand(1, 1, self.n_feats, dtype=torch.float64), torch.randn(1, 1, self.n_feats, dtype=torch.float64))
		outputs = []
		for i, g in enumerate(x):
			out, hidden = self.lstm(g.view(1, 1, -1), hidden)
			out, hidden2 = self.lstm2(g.view(1, 1, -1), hidden2)
			out = self.fcn(out.view(-1))
			outputs.append(2 * out.view(-1))
		return torch.stack(outputs)

## DAGMM Model (ICLR 18)
class DAGMM(nn.Module):
	def __init__(self, feats):
		super(DAGMM, self).__init__()
		self.name = 'DAGMM'
		self.lr = 0.0001
		self.beta = 0.01
		self.n_feats = feats
		self.n_hidden = 16
		self.n_latent = 8
		self.n_window = 5 # DAGMM w_size = 5
		self.n = self.n_feats * self.n_window
		self.n_gmm = self.n_feats * self.n_window
		self.encoder = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.estimate = nn.Sequential(
			nn.Linear(self.n_latent+2, self.n_hidden), nn.Tanh(), nn.Dropout(0.5),
			nn.Linear(self.n_hidden, self.n_gmm), nn.Softmax(dim=1),
		)

	def compute_reconstruction(self, x, x_hat):
		relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
		cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
		return relative_euclidean_distance, cosine_similarity

	def forward(self, x):
		## Encode Decoder
		x = x.view(1, -1)
		z_c = self.encoder(x)
		x_hat = self.decoder(z_c)
		## Compute Reconstructoin
		rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
		z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
		## Estimate
		gamma = self.estimate(z)
		return z_c, x_hat.view(-1), z, gamma.view(-1)

## OmniAnomaly Model (KDD 19)
class OmniAnomaly(nn.Module):
	def __init__(self, feats):
		super(OmniAnomaly, self).__init__()
		self.name = 'OmniAnomaly'
		self.lr = 0.002
		self.beta = 0.01
		self.n_feats = feats
		self.n_hidden = 32
		self.n_latent = 8
		self.lstm = nn.GRU(feats, self.n_hidden, 2)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Flatten(),
			nn.Linear(self.n_hidden, 2*self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
		)

	def forward(self, x, hidden = None):
		hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
		out, hidden = self.lstm(x.view(1, 1, -1), hidden)
		## Encode
		x = self.encoder(out)
		mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
		## Reparameterization trick
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		x = mu + eps*std
		## Decoder
		x = self.decoder(x)
		return x.view(-1), mu.view(-1), logvar.view(-1), hidden

## USAD Model (KDD 20)
class USAD(nn.Module):
	def __init__(self, feats):
		super(USAD, self).__init__()
		self.name = 'USAD'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_hidden = 16
		self.n_latent = 5
		self.n_window = 5 # USAD w_size = 5
		self.n = self.n_feats * self.n_window
		self.encoder = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
		)
		self.decoder1 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.decoder2 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		z = self.encoder(g.view(1,-1))
		## Decoders (Phase 1)
		ae1 = self.decoder1(z)
		ae2 = self.decoder2(z)
		## Encode-Decode (Phase 2)
		ae2ae1 = self.decoder2(self.encoder(ae1))
		return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)

## MSCRED Model (AAAI 19)
class MSCRED(nn.Module):
	def __init__(self, feats):
		super(MSCRED, self).__init__()
		self.name = 'MSCRED'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_window = feats
		self.encoder = nn.ModuleList([
			ConvLSTM(1, 32, (3, 3), 1, True, True, False),
			ConvLSTM(32, 64, (3, 3), 1, True, True, False),
			ConvLSTM(64, 128, (3, 3), 1, True, True, False),
			]
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(128, 64, (3, 3), 1, 1), nn.ReLU(True),
			nn.ConvTranspose2d(64, 32, (3, 3), 1, 1), nn.ReLU(True),
			nn.ConvTranspose2d(32, 1, (3, 3), 1, 1), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		z = g.view(1, 1, self.n_feats, self.n_window)
		for cell in self.encoder:
			_, z = cell(z.view(1, *z.shape))
			z = z[0][0]
		## Decode
		x = self.decoder(z)
		return x.view(-1)

## CAE-M Model (TKDE 21)
class CAE_M(nn.Module):
	def __init__(self, feats):
		super(CAE_M, self).__init__()
		self.name = 'CAE_M'
		self.lr = 0.001
		self.n_feats = feats
		self.n_window = feats
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 8, (3, 3), 1, 1), nn.Sigmoid(),
			nn.Conv2d(8, 16, (3, 3), 1, 1), nn.Sigmoid(),
			nn.Conv2d(16, 32, (3, 3), 1, 1), nn.Sigmoid(),
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(32, 4, (3, 3), 1, 1), nn.Sigmoid(),
			nn.ConvTranspose2d(4, 4, (3, 3), 1, 1), nn.Sigmoid(),
			nn.ConvTranspose2d(4, 1, (3, 3), 1, 1), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		z = g.view(1, 1, self.n_feats, self.n_window)
		z = self.encoder(z)
		## Decode
		x = self.decoder(z)
		return x.view(-1)

## MTAD_GAT Model (ICDM 20)
class MTAD_GAT(nn.Module):
	def __init__(self, feats):
		super(MTAD_GAT, self).__init__()
		self.name = 'MTAD_GAT'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_window = feats
		self.n_hidden = feats * feats
		self.g = dgl.graph((torch.tensor(list(range(1, feats+1))), torch.tensor([0]*feats)))
		self.g = dgl.add_self_loop(self.g)
		self.feature_gat = GATConv(feats, 1, feats)
		self.time_gat = GATConv(feats, 1, feats)
		self.gru = nn.GRU((feats+1)*feats*3, feats*feats, 1)

	def forward(self, data, hidden):
		hidden = torch.rand(1, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
		data = data.view(self.n_window, self.n_feats)
		data_r = torch.cat((torch.zeros(1, self.n_feats), data))
		feat_r = self.feature_gat(self.g, data_r)
		data_t = torch.cat((torch.zeros(1, self.n_feats), data.t()))
		time_r = self.time_gat(self.g, data_t)
		data = torch.cat((torch.zeros(1, self.n_feats), data))
		data = data.view(self.n_window+1, self.n_feats, 1)
		x = torch.cat((data, feat_r, time_r), dim=2).view(1, 1, -1)
		x, h = self.gru(x, hidden)
		return x.view(-1), h

## GDN Model (AAAI 21)
class GDN(nn.Module):
	def __init__(self, feats):
		super(GDN, self).__init__()
		self.name = 'GDN'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_window = 5
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats
		src_ids = np.repeat(np.array(list(range(feats))), feats)
		dst_ids = np.array(list(range(feats))*feats)
		self.g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
		self.g = dgl.add_self_loop(self.g)
		self.feature_gat = GATConv(1, 1, feats)
		self.attention = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_window), nn.Softmax(dim=0),
		)
		self.fcn = nn.Sequential(
			nn.Linear(self.n_feats, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_window), nn.Sigmoid(),
		)

	def forward(self, data):
		# Bahdanau style attention
		att_score = self.attention(data).view(self.n_window, 1)
		data = data.view(self.n_window, self.n_feats)
		data_r = torch.matmul(data.permute(1, 0), att_score)
		# GAT convolution on complete graph
		feat_r = self.feature_gat(self.g, data_r)
		feat_r = feat_r.view(self.n_feats, self.n_feats)
		# Pass through a FCN
		x = self.fcn(feat_r)
		return x.view(-1)

# MAD_GAN (ICANN 19)
class MAD_GAN(nn.Module):
	def __init__(self, feats, n_window=5, lr=0.0001, batch_size=None):
		super(MAD_GAN, self).__init__()
		self.name = 'MAD_GAN'
		self.lr = lr # Use arg
		self.batch = batch_size # Use arg (though not used internally by original MAD_GAN?)
		self.n_feats = feats
		self.n_hidden = 16
		self.n_window = n_window # Use arg
		self.n = self.n_feats * self.n_window
		self.generator = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.discriminator = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
		)

	def forward(self, g):
		# Assuming input g has shape (batch_size, n_window, n_feats) or similar
		# The original code used g.view(1,-1), implying batch_size=1. Need to clarify expected input batching.
		# If batch_size > 1, .view(1, -1) will fail.
		# Let's assume batch_size is the first dimension and adjust.
		batch_size = g.size(0)
		g_flat = g.view(batch_size, -1) # Flatten features per batch item

		## Generate
		z = self.generator(g_flat)
		## Discriminator
		real_score = self.discriminator(g_flat)
		fake_score = self.discriminator(z) # z is already flat

		# Reshape z back? Original returned z.view(-1). If z shape is (B, n_feats*n_window), z.view(-1) flattens completely.
		# Let's return scores per batch item and the reconstructed sequence per batch item.
		# z has shape (batch_size, n), need to reshape to (batch_size, n_window, n_feats)?
		z_reshaped = z.view(batch_size, self.n_window, self.n_feats)
		return z_reshaped, real_score.view(batch_size, -1), fake_score.view(batch_size, -1)

# Proposed Model (VLDB 22)
class TranAD_Basic(nn.Module):
	def __init__(self, feats, n_window=10, lr=1e-3, batch_size=128):
		super(TranAD_Basic, self).__init__()
		self.name = 'TranAD_Basic'
		self.lr = lr # Use arg
		self.batch = batch_size # Use arg
		self.n_feats = feats
		self.n_window = n_window # Use arg
		self.n = self.n_feats * self.n_window
		# Transformer expects (S, B, E) or (B, S, E) if batch_first=True
		# nhead must be divisor of d_model (feats)
		n_head = feats # Assuming feats is divisible by n_head=1, or handle appropriately
		if feats % n_head != 0: n_head = 1 # Simple fallback

		self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
		# Remove batch_first=True
		encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=n_head, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=n_head, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt=None):
		# Inputs shape (batch_size, n_window, n_feats) = (B, S, E)
		if tgt is None:
			 tgt = torch.zeros_like(src)

		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src) # Pos encoding applied to (B, S, E)

		# Permute to (S, B, E) for transformer layers
		src = src.permute(1, 0, 2)
		tgt = tgt.permute(1, 0, 2)

		memory = self.transformer_encoder(src) # Expects (S, B, E)
		x = self.transformer_decoder(tgt, memory) # Expects (T, B, E), (S, B, E)
		# Output x shape (T, B, E)

		# Permute back to (B, T, E)
		x = x.permute(1, 0, 2)

		x = self.fcn(x)
		return x # Output shape (batch_size, n_window, n_feats)

# Proposed Model (FCN) + Self Conditioning + Adversarial + MAML (VLDB 22)
# Note: This uses FCNs, not standard Transformers, despite the name.
class TranAD_Transformer(nn.Module):
	def __init__(self, feats, n_window=10, lr=1e-3, batch_size=128):
		super(TranAD_Transformer, self).__init__()
		self.name = 'TranAD_Transformer'
		self.lr = lr # Use arg
		self.batch = batch_size # Use arg
		self.n_feats = feats
		self.n_hidden = 8
		self.n_window = n_window # Use arg
		# Input size for FCNs: expects concatenated (src, c), flattened. (Batch, 2 * feats * n_window)
		self.n = 2 * self.n_feats * self.n_window
		self.fcn_encoder = nn.Sequential( # Renamed from transformer_encoder
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.ReLU(True))
		self.fcn_decoder1 = nn.Sequential( # Renamed from transformer_decoder1
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
		self.fcn_decoder2 = nn.Sequential( # Renamed from transformer_decoder2
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
		self.fcn_final = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid()) # Renamed from fcn

	def encode(self, src, c):
		# Input src, c: (batch_size, n_window, n_feats)
		batch_size = src.size(0)
		src_c = torch.cat((src, c), dim=2) # Shape (B, T, 2F)
		# Flatten correctly for the FCN: (B, T * 2F)
		src_c_flat = src_c.view(batch_size, -1) # Shape (B, n)
		encoded = self.fcn_encoder(src_c_flat) # Output (B, n)
		return encoded

	def forward(self, src, tgt=None):
		# tgt is not used in this FCN implementation
		batch_size = src.size(0)
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src) # Condition is zeros (B, T, F)
		encoded1 = self.encode(src, c)
		# Decode using FCN decoders
		x1_decoded = self.fcn_decoder1(encoded1) # Output (B, 2 * feats)
		# Reshape and apply final FCN
		# Original reshape was complex: .reshape(-1, 1, 2*self.n_feats).permute(1, 0, 2)
		# -> (B, 1, 2F).permute(1,0,2) -> (1, B, 2F)
		# Simpler reshape for final FCN (assuming it expects B, Features)
		x1_final = self.fcn_final(x1_decoded) # Output (B, feats)

		# Phase 2 - With anomaly scores
		# Condition requires comparing x1_final (B, feats) with src (B, T, F)
		# This comparison is problematic. Original paper might have different architecture.
		# Assuming condition is based on last time step or average?
		# For now, cannot compute c correctly. Using zeros again.
		c2 = torch.zeros_like(src)
		encoded2 = self.encode(src, c2)
		x2_decoded = self.fcn_decoder2(encoded2) # Output (B, 2 * feats)
		x2_final = self.fcn_final(x2_decoded) # Output (B, feats)

		# Return shape (B, feats) - This is different from standard AE (B, T, F)
		# Maybe return the decoded (B, 2*feats) before final FCN? Check loss function.
		# Returning final outputs for now.
		return x1_final, x2_final

# Proposed Model + Self Conditioning + MAML (VLDB 22) - Using actual Transformers
class TranAD_Adversarial(nn.Module):
	def __init__(self, feats, n_window=10, lr=1e-3, batch_size=128):
		super(TranAD_Adversarial, self).__init__()
		self.name = 'TranAD_Adversarial'
		self.lr = lr # Use arg
		self.batch = batch_size # Use arg
		self.n_feats = feats
		self.n_window = n_window # Use arg
		# n_head must be divisor of d_model (2*feats).
		d_model = 2 * feats
		n_head = 2 if feats > 1 and d_model % 2 == 0 else 1 # Ensure divisibility
		if d_model % n_head != 0: n_head = 1 # Fallback if heuristic fails

		self.pos_encoder = PositionalEncoding(d_model, 0.1, self.n_window) # d_model = 2 * feats
		# Remove batch_first=True
		encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sequential(nn.Linear(d_model, feats), nn.Sigmoid()) # Input d_model, output feats

	def encode_decode(self, src, c, tgt_dec_input):
		# Inputs: src(B, S, F), c(B, S, F), tgt_dec_input(B, T, 2F)
		src_c = torch.cat((src, c), dim=2) # Shape: (B, S, 2F)
		src_c = src_c * math.sqrt(src_c.size(-1))
		src_c = self.pos_encoder(src_c) # Applied to (B, S, E)

		# Permute for transformer: (S, B, E)
		src_c = src_c.permute(1, 0, 2)
		tgt_dec_input = tgt_dec_input.permute(1, 0, 2) # Shape (T, B, E)

		memory = self.transformer_encoder(src_c) # Expects (S, B, E), output (S, B, E)
		x = self.transformer_decoder(tgt_dec_input, memory) # Expects (T, B, E), (S, B, E), output (T, B, E)

		# Permute back to (B, T, E)
		x = x.permute(1, 0, 2)

		x = self.fcn(x) # Output shape: (B, T, F)
		return x

	def forward(self, src, tgt=None):
		batch_size, seq_len, _ = src.shape
		d_model = 2 * self.n_feats
		# Decoder input: Shape (B, T, 2*F) - Use zeros
		tgt_dec_input = torch.zeros(batch_size, seq_len, d_model, device=src.device)

		c = torch.zeros_like(src)
		x1 = self.encode_decode(src, c, tgt_dec_input)

		c = (x1 - src) ** 2
		x2 = self.encode_decode(src, c, tgt_dec_input)

		return x1, x2

# Proposed Model + Self Conditioning + MAML (VLDB 22) - SelfConditioning variant
class TranAD_SelfConditioning(nn.Module):
	def __init__(self, feats, n_window=10, lr=1e-3, batch_size=128):
		super(TranAD_SelfConditioning, self).__init__()
		self.name = 'TranAD_SelfConditioning'
		self.lr = lr # Use arg
		self.batch = batch_size # Use arg
		self.n_feats = feats
		self.n_window = n_window # Use arg
		d_model = 2 * feats
		n_head = 2 if feats > 1 and d_model % 2 == 0 else 1
		if d_model % n_head != 0: n_head = 1

		self.pos_encoder = PositionalEncoding(d_model, 0.1, self.n_window)
		# Remove batch_first=True
		encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers1 = TransformerDecoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(nn.Linear(d_model, feats), nn.Sigmoid())

	def encode(self, src, c):
		# Inputs: src(B, S, F), c(B, S, F)
		src_c = torch.cat((src, c), dim=2) # Shape: (B, S, 2F)
		src_c = src_c * math.sqrt(src_c.size(-1))
		src_c = self.pos_encoder(src_c) # Applied to (B, S, E)

		# Permute for encoder: (S, B, E)
		src_c = src_c.permute(1, 0, 2)

		memory = self.transformer_encoder(src_c) # Output shape: (S, B, E)
		return memory

	def forward(self, src, tgt=None):
		batch_size, seq_len, _ = src.shape
		d_model = 2 * self.n_feats
		tgt_dec_input = torch.zeros(batch_size, seq_len, d_model, device=src.device)

		# Permute tgt_dec_input for decoder: (T, B, E)
		tgt_dec_input_permuted = tgt_dec_input.permute(1, 0, 2)

		c1 = torch.zeros_like(src)
		memory1 = self.encode(src, c1) # Output (S, B, E)
		out1 = self.transformer_decoder1(tgt_dec_input_permuted, memory1) # Output (T, B, E)
		# Permute out1 back to (B, T, E)
		out1 = out1.permute(1, 0, 2)
		x1 = self.fcn(out1)

		c2 = (x1 - src) ** 2
		memory2 = self.encode(src, c2) # Output (S, B, E)
		out2 = self.transformer_decoder2(tgt_dec_input_permuted, memory2) # Output (T, B, E)
		# Permute out2 back to (B, T, E)
		out2 = out2.permute(1, 0, 2)
		x2 = self.fcn(out2)

		return x1, x2

# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22) - Full TranAD
class TranAD(nn.Module):
	def __init__(self, feats, n_window=10, lr=1e-3, batch_size=128):
		super(TranAD, self).__init__()
		self.name = 'TranAD'
		self.lr = lr # Use arg
		self.batch = batch_size # Use arg
		self.n_feats = feats
		self.n_window = n_window # Use arg
		d_model = 2 * feats
		n_head = 2 if feats > 1 and d_model % 2 == 0 else 1
		if d_model % n_head != 0: n_head = 1

		self.pos_encoder = PositionalEncoding(d_model, 0.1, self.n_window)
		# Remove batch_first=True
		encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers1 = TransformerDecoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(nn.Linear(d_model, feats), nn.Sigmoid())

	def encode(self, src, c):
		# Inputs: src(B, S, F), c(B, S, F)
		src_c = torch.cat((src, c), dim=2) # Shape: (B, S, 2F)
		src_c = src_c * math.sqrt(src_c.size(-1))
		src_c = self.pos_encoder(src_c) # Applied to (B, S, E)

		# Permute for encoder: (S, B, E)
		src_c = src_c.permute(1, 0, 2)

		memory = self.transformer_encoder(src_c) # Output shape: (S, B, E)
		return memory

	def forward(self, src, tgt=None):
		batch_size, seq_len, _ = src.shape
		d_model = 2 * self.n_feats
		tgt_dec_input = torch.zeros(batch_size, seq_len, d_model, device=src.device)

		# Permute tgt_dec_input for decoder: (T, B, E)
		tgt_dec_input_permuted = tgt_dec_input.permute(1, 0, 2)

		c1 = torch.zeros_like(src)
		memory1 = self.encode(src, c1) # Output (S, B, E)
		out1 = self.transformer_decoder1(tgt_dec_input_permuted, memory1) # Output (T, B, E)
		# Permute out1 back to (B, T, E)
		out1 = out1.permute(1, 0, 2)
		x1 = self.fcn(out1)

		c2 = (x1 - src) ** 2
		memory2 = self.encode(src, c2) # Output (S, B, E)
		out2 = self.transformer_decoder2(tgt_dec_input_permuted, memory2) # Output (T, B, E)
		# Permute out2 back to (B, T, E)
		out2 = out2.permute(1, 0, 2)
		x2 = self.fcn(out2)

		return x1, x2

class Encoder(nn.Module):
	def __init__(self, G_in, G_out, N_layers=1, heads=8):
		super(Encoder, self).__init__()
		self.transf = nn.Transformer(d_model=G_in, 
		nhead=heads, num_encoder_layers=N_layers, num_decoder_layers=N_layers, dim_feedforward=G_in*4, batch_first=True)

	def forward(self, src, tgt):
		return self.transf(src, tgt)

class LSTMAutoencoder(nn.Module):
	def __init__(self, feats, n_window=10, lr=1e-3, batch_size=128, hidden_size=64, num_layers=2, dropout=0.1):
		super(LSTMAutoencoder, self).__init__()
		self.name = 'LSTMAutoencoder'
		self.n_feats = feats
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.n_window = n_window # Use arg
		self.lr = lr # Use arg
		self.batch = batch_size # Use arg

		self.encoder = nn.LSTM(
			input_size=feats,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout
		)
		self.decoder = nn.LSTM(
			input_size=hidden_size, # Decoder input is encoder's hidden state feature size
			hidden_size=hidden_size, # Outputting hidden states
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout
		)
		# Linear layer to map decoder output back to original feature dimension
		self.fc = nn.Linear(hidden_size, feats)

	def forward(self, x):
		# x shape: (batch_size, n_window, n_feats)
		# Encoder
		_, (hidden, cell) = self.encoder(x)
		# hidden shape: (num_layers, batch_size, hidden_size)
		# cell shape: (num_layers, batch_size, hidden_size)

		# Decoder
		# We use the final hidden state of the encoder as the initial hidden state for the decoder.
		# The input to the decoder at each step could be the output from the previous step (teacher forcing during training)
		# or a zero vector, or the last input of the sequence. For simplicity in reconstruction,
		# we can feed a sequence of zeros or the encoded representation repeatedly.
		# Here, let's use the last hidden state as context and decode step-by-step implicitly
		# by feeding a dummy input sequence of the same length as the input window.
		# A common AE approach is to repeat the context vector.

		# Option 1: Repeat last hidden state (or context vector) as input
		# We take the hidden state from the last layer
		# Ensure n_window is available for repeat
		if not hasattr(self, 'n_window') or self.n_window is None:
			# Fallback if n_window wasn't set (e.g., if loaded from old checkpoint)
			# Infer from input shape if possible, else use a default like 10
			self.n_window = x.size(1) if len(x.shape) > 1 else 10

		context = hidden[-1].unsqueeze(1).repeat(1, self.n_window, 1)
		# context shape: (batch_size, n_window, hidden_size)

		decoder_output, _ = self.decoder(context, (hidden, cell)) # Use encoder's final state
		# decoder_output shape: (batch_size, n_window, hidden_size)

		# Map decoder output to original feature space
		reconstruction = self.fc(decoder_output)
		# reconstruction shape: (batch_size, n_window, n_feats)

		# Return only the reconstruction of the last element for consistency with some other models if needed
		# Or return the whole sequence reconstruction depending on how loss is calculated in main.py
		# The generic backprop loop currently assumes the output shape matches input shape for basic models.
		return reconstruction

class VanillaTransformer(nn.Module):
	def __init__(self, feats, n_window=10, lr=1e-3, batch_size=128, n_head=8, n_layers=2, dropout=0.1, d_model=None):
		super(VanillaTransformer, self).__init__()
		self.name = 'VanillaTransformer'
		self.n_feats = feats
		self.n_window = n_window # Use arg
		self.lr = lr # Use arg
		self.batch = batch_size # Use arg

		# If d_model is not specified, use feats or a default value like 512?
		# Using feats might be too small if feats is small.
		# Let's make it configurable, defaulting to feats but maybe larger preferred.
		self.d_model = d_model if d_model else feats * 4 # Example: scale feats
		self.n_head = n_head
		self.n_layers = n_layers
		if self.d_model % n_head != 0:
			print(f"Warning: d_model ({self.d_model}) not divisible by n_head ({n_head}). Adjusting n_head to 1.")
			self.n_head = 1

		# Input embedding layer (optional but good practice)
		self.input_embed = nn.Linear(feats, self.d_model)

		self.pos_encoder = PositionalEncoding(self.d_model, dropout, max_len=self.n_window + 1) # From TranAD

		encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head, dim_feedforward=self.d_model * 4, dropout=dropout)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

		# Output layer to map back to feature dimension
		self.output_layer = nn.Linear(self.d_model, feats)

		self.init_weights()

	def init_weights(self):
		initrange = 0.1
		self.input_embed.weight.data.uniform_(-initrange, initrange)
		self.output_layer.bias.data.zero_()
		self.output_layer.weight.data.uniform_(-initrange, initrange)

	def forward(self, src):
		# src shape: (B, S, E)
		if not hasattr(self, 'n_window') or self.n_window is None:
			# Check if n_window is defined and potentially update PositionalEncoding
			# This might be needed if the model is loaded without saving n_window
			# or if the input sequence length changes dynamically
			# Example (if needed):
			# if src.size(1) != self.n_window:
			# 	self.n_window = src.size(1)
			# 	self.pos_encoder = PositionalEncoding(self.d_model, self.dropout_rate, max_len=self.n_window + 1).to(src.device)
			pass # Placeholder if no immediate action is required

		src = self.input_embed(src) * math.sqrt(self.d_model)
		src = self.pos_encoder(src) # Applied to (B, S, E)

		# Permute for encoder: (S, B, E)
		src = src.permute(1, 0, 2)

		# For a standard Transformer AE, the target (tgt) should be the same as the source (src)
		# The standard nn.Transformer expects tgt for the decoder part.
		# When using it as an autoencoder, we often feed the src as tgt.
		# The model will use causal masking internally if tgt_mask is provided,
		# but for AE, we typically don't need causal masking for the decoder input.
		# If a mask is needed (e.g., padding mask), it should be generated and passed.
		# For simplicity, assuming no padding mask needed here.
		output = self.transformer_encoder(src=src, tgt=src) # Output (S, B, E)

		# Permute back to (B, S, E)
		output = output.permute(1, 0, 2)

		reconstruction = self.output_layer(output)
		return reconstruction

class GRUAutoencoder(nn.Module):
	def __init__(self, feats, n_window=10, lr=1e-3, batch_size=128, hidden_size=64, num_layers=2, dropout=0.1):
		super(GRUAutoencoder, self).__init__()
		self.name = 'GRUAutoencoder'
		self.n_feats = feats
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.n_window = n_window # Use arg
		self.lr = lr # Use arg
		self.batch = batch_size # Use arg

		self.encoder = nn.GRU(
			input_size=feats,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout
		)
		self.decoder = nn.GRU(
			input_size=hidden_size, # Use hidden_size as input to decoder
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout
		)
		self.fc = nn.Linear(hidden_size, feats)

	def forward(self, x):
		# x shape: (batch_size, n_window, n_feats)
		# Encoder
		_, hidden = self.encoder(x) # GRU returns only final hidden state
		# hidden shape: (num_layers, batch_size, hidden_size)

		# Decoder
		# Use encoder's final hidden state for decoder initial state
		# Repeat context vector as input for decoder
		if not hasattr(self, 'n_window') or self.n_window is None:
			self.n_window = x.size(1) if len(x.shape) > 1 else 10
		context = hidden[-1].unsqueeze(1).repeat(1, self.n_window, 1)
		decoder_output, _ = self.decoder(context, hidden) # Use encoder's final state

		# Map decoder output to original feature space
		reconstruction = self.fc(decoder_output)
		return reconstruction

class ConvLSTMAutoencoder(nn.Module):
	def __init__(self, feats, n_window=10, lr=1e-3, batch_size=128, hidden_dims=[16, 32], kernel_sizes=[(3,3), (3,3)], num_layers=2, dropout=0.1):
		super(ConvLSTMAutoencoder, self).__init__()
		self.name = 'ConvLSTMAutoencoder'
		self.n_feats = feats
		self.n_window = n_window # Use arg
		self.lr = lr # Use arg
		self.batch = batch_size # Use arg
		
		# Input requires shape (B, T, C, H, W). Here C=1, H=feats, W=1? Or maybe H=1, W=feats?
		# Let's assume (B, T, C=1, H=feats, W=1) for input
		# Kernel sizes should match num_layers. Hidden dims should match num_layers.
		if isinstance(hidden_dims, int): hidden_dims = [hidden_dims] * num_layers
		if isinstance(kernel_sizes, tuple): kernel_sizes = [kernel_sizes] * num_layers
		assert len(hidden_dims) == num_layers, "Length of hidden_dims must match num_layers"
		assert len(kernel_sizes) == num_layers, "Length of kernel_sizes must match num_layers"

		self.hidden_dims = hidden_dims
		self.kernel_sizes = kernel_sizes
		self.num_layers = num_layers

		# Encoder uses ConvLSTM cells
		self.encoder = ConvLSTM(input_dim=1, # Channels = 1
								hidden_dim=hidden_dims,
								kernel_size=kernel_sizes,
								num_layers=num_layers,
								batch_first=True,
								bias=True,
								return_all_layers=True) # Force return_all_layers=True
		
		# Decoder needs to map back. Can use ConvLSTM or ConvTranspose layers.
		# Using ConvLSTM decoder mirrored to encoder
		decoder_hidden_dims = hidden_dims[::-1] # Reverse hidden dims
		# Input dim for decoder is last encoder hidden dim
		decoder_input_dim = hidden_dims[-1] 
		# Output dim needs to be 1 channel eventually
		# We need ConvLSTM layers followed by a final Conv2d to map to 1 channel.

		self.decoder_convlstm = ConvLSTM(input_dim=decoder_input_dim, # Input is last encoder hidden dim
										 hidden_dim=decoder_hidden_dims,
										 kernel_size=kernel_sizes[::-1], # Use reversed kernels?
										 num_layers=num_layers,
										 batch_first=True,
										 bias=True,
										 return_all_layers=True) # Force return_all_layers=True
		
		# Final Conv2d to map last decoder hidden state channels to 1 channel output
		self.final_conv = nn.Conv2d(in_channels=decoder_hidden_dims[-1], # Last hidden dim of decoder
									 out_channels=1, # Output 1 channel
									 kernel_size=(1, 1), # Keep H, W same
									 padding=0)


	def forward(self, x):
		# Input x: (batch_size, n_window, n_feats)
		# Reshape to (B, T, C, H, W) = (batch_size, n_window, 1, n_feats, 1)
		b, t, f = x.shape
		x_reshaped = x.view(b, t, 1, f, 1)

		# Encoder
		encoder_outputs, encoder_last_states = self.encoder(x_reshaped)
		# encoder_outputs is a list of outputs for each layer. We need the last layer's output sequence.
		# encoder_last_states is a list of (h, c) tuples for each layer's final state.
		
		last_layer_encoder_output = encoder_outputs[-1] # Shape (B, T, C_hidden, H, W)
		last_layer_final_state = encoder_last_states[-1] # Tuple (h, c)

		# Decoder
		# Input to decoder LSTM: Last encoder hidden state repeated?
		# Let's use the output sequence of the last encoder layer as input sequence to the first decoder layer.
		# The initial hidden state for the decoder layers should be the final hidden state from corresponding encoder layer (or reversed).
		
		# Need initial hidden state for decoder ConvLSTM. Use encoder's last state, possibly reversed.
		decoder_init_states = [(h.clone(), c.clone()) for h,c in encoder_last_states[::-1]] # Reversed states

		# Input sequence for decoder ConvLSTM: Use the output sequence of the last encoder layer
		decoder_input_seq = last_layer_encoder_output

		decoder_outputs, _ = self.decoder_convlstm(decoder_input_seq, decoder_init_states)
		
		# We need the output of the last decoder layer
		last_layer_decoder_output = decoder_outputs[-1] # Shape (B, T, C_dec_hidden_last, H, W)

		# Apply final conv layer to get (B, T, 1, H, W)
		reconstruction_reshaped = self.final_conv(last_layer_decoder_output.view(-1, *last_layer_decoder_output.shape[2:])) # Need to apply conv per time step?
		# Reshape output from (B*T, 1, H, W) back to (B, T, 1, H, W)
		# Alternative: Apply 3D Conv? Simpler: Apply 2D Conv to each time step's output.
		
		# Apply final_conv across time dimension (permute, view, conv, view, permute)
		b_dec, t_dec, c_dec, h_dec, w_dec = last_layer_decoder_output.shape
		conv_input = last_layer_decoder_output.permute(0, 2, 1, 3, 4).contiguous().view(b_dec, c_dec, t_dec * h_dec, w_dec) # Example reshape, check if correct
		# This reshape seems wrong. Let's apply the final conv per time step.
		
		output_seq = []
		for time_step in range(last_layer_decoder_output.size(1)):
			step_output = self.final_conv(last_layer_decoder_output[:, time_step, :, :, :]) # (B, 1, H, W)
			output_seq.append(step_output)
		
		reconstruction_reshaped = torch.stack(output_seq, dim=1) # (B, T, 1, H, W) = (B, T, 1, feats, 1)

		# Reshape back to (batch_size, n_window, n_feats)
		reconstruction = reconstruction_reshaped.view(b, t, f)
		return reconstruction

class ConvGRUAutoencoder(nn.Module):
	def __init__(self, feats, n_window=10, lr=1e-3, batch_size=128, hidden_dims=[16, 32], kernel_sizes=[(3,3), (3,3)], num_layers=2, dropout=0.1): # dropout not used by ConvGRU? Check impl.
		super(ConvGRUAutoencoder, self).__init__()
		self.name = 'ConvGRUAutoencoder'
		self.n_feats = feats
		self.n_window = n_window
		self.lr = lr
		self.batch = batch_size

		if isinstance(hidden_dims, int): hidden_dims = [hidden_dims] * num_layers
		if isinstance(kernel_sizes, tuple): kernel_sizes = [kernel_sizes] * num_layers
		assert len(hidden_dims) == num_layers
		assert len(kernel_sizes) == num_layers

		self.hidden_dims = hidden_dims
		self.kernel_sizes = kernel_sizes
		self.num_layers = num_layers

		# Encoder uses ConvGRU cells from dlutils
		self.encoder = ConvGRU(input_dim=1,
							   hidden_dim=hidden_dims,
							   kernel_size=kernel_sizes,
							   num_layers=num_layers,
							   batch_first=True,
							   bias=True,
							   return_all_layers=True) # Force return_all_layers=True

		# Decoder mirrored to encoder
		decoder_hidden_dims = hidden_dims[::-1]
		decoder_input_dim = hidden_dims[-1]

		self.decoder_convgru = ConvGRU(input_dim=decoder_input_dim,
									   hidden_dim=decoder_hidden_dims,
									   kernel_size=kernel_sizes[::-1],
									   num_layers=num_layers,
									   batch_first=True,
									   bias=True,
									   return_all_layers=True) # Force return_all_layers=True

		self.final_conv = nn.Conv2d(in_channels=decoder_hidden_dims[-1],
									 out_channels=1,
									 kernel_size=(1, 1),
									 padding=0)

	def forward(self, x):
		# Input x: (B, T, F)
		b, t, f = x.shape
		x_reshaped = x.view(b, t, 1, f, 1)

		# Encoder
		encoder_outputs, encoder_last_states = self.encoder(x_reshaped)
		last_layer_encoder_output = encoder_outputs[-1] # (B, T, C_hidden, H, W)
		# GRU last_states is just a list of h tensors per layer

		# Decoder
		decoder_init_states = [h.clone() for h in encoder_last_states[::-1]] # Reversed states
		decoder_input_seq = last_layer_encoder_output
		decoder_outputs, _ = self.decoder_convgru(decoder_input_seq, decoder_init_states)
		last_layer_decoder_output = decoder_outputs[-1] # (B, T, C_dec_hidden_last, H, W)

		# Apply final conv layer per time step
		output_seq = []
		for time_step in range(last_layer_decoder_output.size(1)):
			step_output = self.final_conv(last_layer_decoder_output[:, time_step, :, :, :]) # (B, 1, H, W)
			output_seq.append(step_output)

		reconstruction_reshaped = torch.stack(output_seq, dim=1) # (B, T, 1, H, W) -> (B, T, 1, feats, 1)

		# Reshape back to (B, T, F)
		reconstruction = reconstruction_reshaped.view(b, t, f)
		return reconstruction

# --- Placeholder for Attention Variants ---
# ConvLSTMAttentionAutoencoder and ConvGRUAttentionAutoencoder would require
# incorporating an attention mechanism, likely between encoder and decoder,
# or within the decoder. This requires a more complex design than the simple AE above.
# For now, let's define them as inheriting from the non-attention versions
# and add the parameters, but the forward pass won't use attention yet.

class ConvLSTMAttentionAutoencoder(ConvLSTMAutoencoder):
	def __init__(self, feats, n_window=10, lr=1e-3, batch_size=128, hidden_dims=[16, 32], kernel_sizes=[(3,3), (3,3)], num_layers=2, att_heads=4, dropout=0.1):
		# Call grandparent nn.Module init directly, bypassing ConvLSTMAutoencoder.__init__
		super(ConvLSTMAutoencoder, self).__init__() 
		self.name = 'ConvLSTMAttentionAutoencoder'
		self.n_feats = feats
		self.n_window = n_window
		self.lr = lr
		self.batch = batch_size
		self.att_heads = att_heads

		if isinstance(hidden_dims, int): hidden_dims = [hidden_dims] * num_layers
		if isinstance(kernel_sizes, tuple): kernel_sizes = [kernel_sizes] * num_layers
		assert len(hidden_dims) == num_layers
		assert len(kernel_sizes) == num_layers

		self.hidden_dims = hidden_dims
		self.kernel_sizes = kernel_sizes
		self.num_layers = num_layers
		self.dropout_rate = dropout # Store dropout rate

		# --- Encoder Definition ---
		self.encoder = ConvLSTM(input_dim=1, # Input channels = 1
								hidden_dim=hidden_dims, 
								kernel_size=kernel_sizes, 
								num_layers=num_layers,
								batch_first=True,
								bias=True,
								return_all_layers=True) # Removed dropout

		# --- Attention Layer ---
		encoder_last_hidden_dim = hidden_dims[-1]
		# Assuming H=feats, W=1 for input x_reshaped = x.view(b, t, 1, f, 1)
		self.embed_dim = encoder_last_hidden_dim * self.n_feats # C_enc * H * W = C_enc * feats * 1
		# Ensure embed_dim is divisible by att_heads
		if self.embed_dim % self.att_heads != 0:
			# Adjust embed_dim or att_heads. Simplest: adjust heads if possible.
			# Find the largest divisor of embed_dim <= original att_heads
			valid_heads = [h for h in range(1, self.att_heads + 1) if self.embed_dim % h == 0]
			if not valid_heads: 
				print(f"Warning: embed_dim ({self.embed_dim}) not divisible by any #heads <= {self.att_heads}. Setting heads=1.")
				self.att_heads = 1
			else:
				new_heads = max(valid_heads)
				if new_heads != self.att_heads:
					print(f"Warning: embed_dim ({self.embed_dim}) not divisible by att_heads ({self.att_heads}). Adjusting heads to {new_heads}.")
					self.att_heads = new_heads

		self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim,
								num_heads=self.att_heads,
								batch_first=True,
								dropout=self.dropout_rate) # Add dropout to attention

		# --- Decoder Definition (Modified Input Dim) ---
		decoder_hidden_dims = hidden_dims[::-1] 
		# Input to the first decoder layer includes concatenated context
		decoder_input_dim = encoder_last_hidden_dim * 2 # Original encoder output channels + context channels

		self.decoder_convlstm = ConvLSTM(input_dim=decoder_input_dim,
									   hidden_dim=decoder_hidden_dims,
									   kernel_size=kernel_sizes[::-1], 
									   num_layers=num_layers,
									   batch_first=True,
									   bias=True,
									   return_all_layers=True) # Removed dropout

		# --- Final Conv Layer ---
		self.final_conv = nn.Conv2d(in_channels=decoder_hidden_dims[-1], # Channels from last decoder layer
									 out_channels=1, 
									 kernel_size=(1, 1), 
									 padding=0)

	# --- Reimplement forward pass WITH attention ---
	def forward(self, x):
		b, t, f = x.shape
		# Reshape: (B, T, F) -> (B, T, C=1, H=f, W=1)
		x_reshaped = x.view(b, t, 1, f, 1)

		# Encoder
		encoder_outputs, encoder_last_states = self.encoder(x_reshaped)
		# encoder_outputs: List layer outputs [(B, T, C_l, H, W)]
		# encoder_last_states: List layer final states [(h_n, c_n)], h_n shape (B, C_l, H, W)

		last_layer_encoder_output = encoder_outputs[-1] # (B, T, C_enc, H, W)
		enc_last_h, enc_last_c = encoder_last_states[-1]   # (B, C_enc, H, W)
		C_enc = self.hidden_dims[-1]
		H, W = f, 1

		# Attention Mechanism
		# Query: Final encoder hidden state (h)
		# Key/Value: Encoder output sequence from last layer
		query = enc_last_h.view(b, 1, self.embed_dim)                # (B, 1, embed_dim)
		memory = last_layer_encoder_output.view(b, t, self.embed_dim) # (B, T, embed_dim)

		# Apply attention
		context, attn_weights = self.attention(query=query, key=memory, value=memory)
		# context shape: (B, 1, embed_dim)

		# Reshape context back to spatial dims for ConvLSTM
		context = context.view(b, 1, C_enc, H, W) # (B, 1, C_enc, H, W)

		# Repeat context for all time steps 
		context_repeated = context.repeat(1, t, 1, 1, 1) # (B, T, C_enc, H, W)

		# Prepare Decoder Input
		# Concatenate original encoder output sequence with the context
		decoder_input_seq = torch.cat((last_layer_encoder_output, context_repeated), dim=2) # Concat on channel dim
		# Shape: (B, T, C_enc * 2, H, W)

		# Prepare Decoder Initial States (use original encoder final states, reversed)
		decoder_init_states = [(h.clone(), c.clone()) for h, c in encoder_last_states[::-1]]

		# Decoder
		decoder_outputs, _ = self.decoder_convlstm(decoder_input_seq, decoder_init_states)
		# decoder_outputs: List layer outputs [(B, T, C_dec_l, H, W)]

		last_layer_decoder_output = decoder_outputs[-1] # (B, T, C_dec_last, H, W)

		# Apply final conv layer per time step
		output_seq = []
		for time_step in range(last_layer_decoder_output.size(1)):
			step_output = self.final_conv(last_layer_decoder_output[:, time_step, :, :, :]) # (B, 1, H, W)
			output_seq.append(step_output)
		
		reconstruction_reshaped = torch.stack(output_seq, dim=1) # (B, T, 1, H, W)

		# Reshape back to (B, T, F)
		reconstruction = reconstruction_reshaped.view(b, t, f)
		return reconstruction

class ConvGRUAttentionAutoencoder(ConvGRUAutoencoder):
	def __init__(self, feats, n_window=10, lr=1e-3, batch_size=128, hidden_dims=[16, 32], kernel_sizes=[(3,3), (3,3)], num_layers=2, att_heads=4, dropout=0.1):
		# Call grandparent nn.Module init
		super(ConvGRUAutoencoder, self).__init__() 
		self.name = 'ConvGRUAttentionAutoencoder'
		self.n_feats = feats
		self.n_window = n_window
		self.lr = lr
		self.batch = batch_size
		self.att_heads = att_heads

		if isinstance(hidden_dims, int): hidden_dims = [hidden_dims] * num_layers
		if isinstance(kernel_sizes, tuple): kernel_sizes = [kernel_sizes] * num_layers
		assert len(hidden_dims) == num_layers
		assert len(kernel_sizes) == num_layers

		self.hidden_dims = hidden_dims
		self.kernel_sizes = kernel_sizes
		self.num_layers = num_layers
		self.dropout_rate = dropout # Store dropout rate

		# --- Encoder Definition ---
		self.encoder = ConvGRU(input_dim=1, 
							   hidden_dim=hidden_dims,
							   kernel_size=kernel_sizes,
							   num_layers=num_layers,
							   batch_first=True,
							   bias=True,
							   return_all_layers=True) # ConvGRU from dlutils might not support dropout

		# --- Attention Layer ---
		encoder_last_hidden_dim = hidden_dims[-1]
		self.embed_dim = encoder_last_hidden_dim * self.n_feats # C_enc * H * W
		# Ensure embed_dim is divisible by att_heads
		if self.embed_dim % self.att_heads != 0:
			valid_heads = [h for h in range(1, self.att_heads + 1) if self.embed_dim % h == 0]
			if not valid_heads: 
				print(f"Warning: embed_dim ({self.embed_dim}) not divisible by any #heads <= {self.att_heads}. Setting heads=1.")
				self.att_heads = 1
			else:
				new_heads = max(valid_heads)
				if new_heads != self.att_heads:
					print(f"Warning: embed_dim ({self.embed_dim}) not divisible by att_heads ({self.att_heads}). Adjusting heads to {new_heads}.")
					self.att_heads = new_heads
		
		self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim,
								num_heads=self.att_heads,
								batch_first=True,
								dropout=self.dropout_rate)

		# --- Decoder Definition (Modified Input Dim) ---
		decoder_hidden_dims = hidden_dims[::-1]
		decoder_input_dim = encoder_last_hidden_dim * 2

		self.decoder_convgru = ConvGRU(input_dim=decoder_input_dim,
									   hidden_dim=decoder_hidden_dims,
									   kernel_size=kernel_sizes[::-1],
									   num_layers=num_layers,
									   batch_first=True,
									   bias=True,
									   return_all_layers=True)

		# --- Final Conv Layer ---
		self.final_conv = nn.Conv2d(in_channels=decoder_hidden_dims[-1],
									out_channels=1,
									kernel_size=(1, 1),
									padding=0)

	# --- Reimplement forward pass WITH attention ---
	def forward(self, x):
		b, t, f = x.shape
		x_reshaped = x.view(b, t, 1, f, 1) # (B, T, C=1, H=f, W=1)

		# Encoder
		encoder_outputs, encoder_last_states = self.encoder(x_reshaped)
		# encoder_outputs: List layer outputs [(B, T, C_l, H, W)]
		# encoder_last_states: List layer final states [h_n], h_n shape (B, C_l, H, W)

		last_layer_encoder_output = encoder_outputs[-1] # (B, T, C_enc, H, W)
		enc_last_h = encoder_last_states[-1]           # (B, C_enc, H, W)
		C_enc = self.hidden_dims[-1]
		H, W = f, 1

		# Attention Mechanism
		query = enc_last_h.view(b, 1, self.embed_dim)                # (B, 1, embed_dim)
		memory = last_layer_encoder_output.view(b, t, self.embed_dim) # (B, T, embed_dim)

		# Apply attention
		context, attn_weights = self.attention(query=query, key=memory, value=memory)
		# context shape: (B, 1, embed_dim)

		# Reshape context back to spatial dims 
		context = context.view(b, 1, C_enc, H, W) # (B, 1, C_enc, H, W)

		# Repeat context for all time steps
		context_repeated = context.repeat(1, t, 1, 1, 1) # (B, T, C_enc, H, W)

		# Prepare Decoder Input
		decoder_input_seq = torch.cat((last_layer_encoder_output, context_repeated), dim=2) # Concat on channel dim
		# Shape: (B, T, C_enc * 2, H, W)

		# Prepare Decoder Initial States
		decoder_init_states = [h.clone() for h in encoder_last_states[::-1]] # List of h tensors

		# Decoder
		decoder_outputs, _ = self.decoder_convgru(decoder_input_seq, decoder_init_states)
		# decoder_outputs: List layer outputs [(B, T, C_dec_l, H, W)]

		last_layer_decoder_output = decoder_outputs[-1] # (B, T, C_dec_last, H, W)

		# Apply final conv layer per time step
		output_seq = []
		for time_step in range(last_layer_decoder_output.size(1)):
			step_output = self.final_conv(last_layer_decoder_output[:, time_step, :, :, :]) # (B, 1, H, W)
			output_seq.append(step_output)
		
		reconstruction_reshaped = torch.stack(output_seq, dim=1) # (B, T, 1, H, W)

		# Reshape back to (B, T, F)
		reconstruction = reconstruction_reshaped.view(b, t, f)
		return reconstruction

# ... Need to add TransformerEncoderDecoder class definition ...
# Assuming it should also be an AE model using Transformer layers

class TransformerEncoderDecoder(nn.Module):
	def __init__(self, feats, n_window=10, lr=1e-3, batch_size=128, n_head=8, enc_layers=2, dec_layers=2, dropout=0.1, d_model=None):
		super(TransformerEncoderDecoder, self).__init__()
		self.name = 'TransformerEncoderDecoder'
		self.n_feats = feats
		self.n_window = n_window
		self.lr = lr
		self.batch = batch_size

		self.d_model = d_model if d_model else feats * 4
		self.n_head = n_head
		if self.d_model % n_head != 0:
			print(f"Warning: d_model ({self.d_model}) not divisible by n_head ({n_head}). Adjusting n_head to 1.")
			self.n_head = 1
			
		self.input_embed = nn.Linear(feats, self.d_model)
		self.pos_encoder = PositionalEncoding(self.d_model, dropout, max_len=self.n_window + 1)

		# Using standard PyTorch Transformer - remove batch_first=True
		self.transformer = nn.Transformer(
			d_model=self.d_model,
			nhead=self.n_head,
			num_encoder_layers=enc_layers,
			num_decoder_layers=dec_layers,
			dim_feedforward=self.d_model * 4,
			dropout=dropout
			# Removed batch_first=True
		)
		self.output_layer = nn.Linear(self.d_model, feats)
		self.init_weights()

	def init_weights(self):
		initrange = 0.1
		self.input_embed.weight.data.uniform_(-initrange, initrange)
		self.output_layer.bias.data.zero_()
		self.output_layer.weight.data.uniform_(-initrange, initrange)

	def _generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

	def forward(self, src):
		# src shape: (B, S, E)
		if not hasattr(self, 'n_window') or self.n_window is None:
			# Check if n_window is defined and potentially update PositionalEncoding
			# This might be needed if the model is loaded without saving n_window
			# or if the input sequence length changes dynamically
			# Example (if needed):
			# if src.size(1) != self.n_window:
			# 	self.n_window = src.size(1)
			# 	self.pos_encoder = PositionalEncoding(self.d_model, self.dropout_rate, max_len=self.n_window + 1).to(src.device)
			pass # Placeholder if no immediate action is required

		src = self.input_embed(src) * math.sqrt(self.d_model)
		src = self.pos_encoder(src) # Applied to (B, S, E)

		# Permute for encoder: (S, B, E)
		src = src.permute(1, 0, 2)

		# For a standard Transformer AE, the target (tgt) should be the same as the source (src)
		# The standard nn.Transformer expects tgt for the decoder part.
		# When using it as an autoencoder, we often feed the src as tgt.
		# The model will use causal masking internally if tgt_mask is provided,
		# but for AE, we typically don't need causal masking for the decoder input.
		# If a mask is needed (e.g., padding mask), it should be generated and passed.
		# For simplicity, assuming no padding mask needed here.
		output = self.transformer(src=src, tgt=src) # Output (S, B, E)

		# Permute back to (B, S, E)
		output = output.permute(1, 0, 2)

		reconstruction = self.output_layer(output)
		return reconstruction
