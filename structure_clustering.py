from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from umap import umap_ as umap
import numpy as np
import ase
import os

path = "platelet_11/"
atoms = ase.io.read(os.path.join(path, "structure_mod.xyz"))




# 1. SOAP-Deskriptor
soap = SOAP(
    species=["W", "S"],
    r_cut=4.5,       # ~2. Nachbarschale
    n_max=9,
    l_max=9,
    periodic=False,
    sparse=False
)

features = soap.create(atoms)  # shape: (N_atoms, N_features)

# Nur W-Atome klassifizieren
w_mask = np.array([a.symbol == 'W' for a in atoms])
X = features[w_mask]

# 2. Normalisieren
X = StandardScaler().fit_transform(X)

# 3. Dimensionsreduktion
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
X_2d = reducer.fit_transform(X)

# 4. Clustering
# HDBSCAN: findet Cluster automatisch, markiert Ausreißer als -1
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
labels = clusterer.fit_predict(X_2d)