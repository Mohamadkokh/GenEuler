# config.py
import os, torch

# use CUDA only if both the env var is set *and* torch.cuda is actually available
USE_CUDA = os.getenv("USE_CUDA", "0") == "1"
DEVICE   = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"

# SAC hyperparameters.
SAC_CONFIG = {
    "density_threshold": 0.5,
    "max_add_radius": 1.0,
    "actor_lr": 1e-5,
    "critic_lr": 1e-5,
    "entropy_coeff": 0.2,
    "discount_factor": 0.99,
    "tau": 0.005,
    "replay_buffer_size": 1000000,
    "batch_size": 1,
    "max_epochs": 100,
    "max_steps": 1000,
    "updates_per_step": 3, 
    # weight for rewarding the agent anytime it adds material
    "addition_reward_coeff": 0.1,
    "min_fallback_count": 10000,  # minimum number of fallback steps before stopping
    "max_add_candidates": 150,
    "min_add_norm": 1e-3,  # minimum norm for a point to be considered for addition
    "weight_decay": 1e-4,
}

# # GNN model configuration.
# GNN_CONFIG = {
#   "input_dim": 10,              # your feature count
#   "shared_num_layers": 8,       # depth of the shared GNN
#   "shared_hidden_dim": 164,      # width of each shared GNN layer
#   "branch_num_layers": 5,       # depth of each branch MLP
#   "branch_hidden_dim": 132,      # width of each branch MLP
#   "output_dim": 16,             # your branch output dim
#   "fusion_dim": 32,             # post‑fusion hidden dim
# }

# # GNN model configuration.
# GNN_CONFIG = {
#   "input_dim": 10,            
#   "shared_num_layers": 8,    
#   "shared_hidden_dim": 164,      
#   "branch_num_layers": 5,       
#   "branch_hidden_dim": 132,      
#   "output_dim": 16,            
#   "fusion_dim": 32,             
# }

# # Lightweight (fast inference, <2 GB):
GNN_CONFIG = {
   "input_dim":       10,   # same number of per-node features
   "shared_num_layers": 4,  # fewer GCN layers
   "shared_hidden_dim": 128,  # much slimmer hidden channels
   "branch_num_layers": 2,  # shallow MLP heads
   "branch_hidden_dim":  64,  # slimmer branch channels
   "output_dim":         32,  # lower-dim action embedding
   "fusion_dim":         64,  # smaller post-fusion layer
 }

# Mid‑range (medium speed, ≲6 GB)::
#GNN_CONFIG = {
 # "input_dim": 10,
 # "shared_num_layers": 6,
 # "shared_hidden_dim": 512,
 # "branch_num_layers": 3,
 # "branch_hidden_dim": 256,
 # "output_dim": 64,
 # "fusion_dim": 128,
#}

# # Heavy Slower (slower, up to ≲15 GB):
# GNN_CONFIG = {
#   "input_dim": 10,
#   "shared_num_layers": 8,
#   "shared_hidden_dim": 1024,
#   "branch_num_layers": 4,
#   "branch_hidden_dim": 512,
#   "output_dim": 128,
#   "fusion_dim": 256,
# }

# Training configuration.
TRAINING_CONFIG = {
    "num_epochs": 100,
    "log_interval": 10,
    "checkpoint_dir": "./checkpoints",
}

# Shared assets paths (all JSON files are stored in the shared_assets volume mounted at /app/src/assets)
DATA_PATHS = {
    "merged_data": "/app/src/assets",
    "mesh_data": "/app/src/assets",
}

# Add the bounding box used in the shared environment
BOUNDING_BOX = {
    "x": [-50, 50],
    "y": [0, 130],
    "z": [-130, 130]
}

# Attachment point – material should not be added near this point.
ATTACHMENT_POSITION = [0, 130, -110]
TOLERANCE = 5.0

# Minimum allowed vertices in the environment.
MIN_ALLOWED_VERTICES = 4  # Lower this if 6 is too strict for your use case.
