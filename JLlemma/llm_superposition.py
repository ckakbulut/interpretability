import numpy as np

# Superposition simulation

no_features = 100
no_neurons = 10

# Generate high-dimensional feature representations and projection matrix (equivalent to (W)eight matrix in LLM)
ft_vectors = np.random.randn(no_features, no_features)
proj_matrix = np.random.randn(no_features, no_neurons)

activations = ft_vectors @ proj_matrix # (100, 100) x (100, 10) = (100, 10)


