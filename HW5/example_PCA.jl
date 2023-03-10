using DelimitedFiles
include("PCA.jl")

# Load data
dataTable = readdlm("animals.csv",',')
X = float(real(dataTable[2:end,2:end]))
animal_names = dataTable[2:end, 1]
(n,d) = size(X)

# Plot matrix as image
using Plots
# heatmap(X)

# Use PCA to shrink feature dimension to 2 (k=2)
pca_model = PCA(X, 2)

# Compress the data
X_pca = pca_model.compress(X)

# Plot the scatter plot
scatter(X_pca[:, 1], X_pca[:, 2], legend=false)
title!("Scatter Plot of PCA-Compressed Data")
annotate!(X_pca[:, 1], X_pca[:, 2].-0.1, animal_names, annotationfontsize=8)

