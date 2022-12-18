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
pca_model = PCA(X, 11)

# Compress the data
feat_means = mean(X, dims=1)
X_pca = pca_model.compress(X)

# Compute feature means and center X
X_c = X - repeat(feat_means,n,1)

# Find explained variance
explained_var = 1 - (norm(pca_model.expand(X_pca) - repeat(feat_means,n,1) - X_c)^2 / norm(X_c)^2)
print("The explained variance is:  \n", explained_var * 100)

# Plot the scatter plot
scatter(X_pca[:, 1], X_pca[:, 2], legend=false)
title!("Scatter Plot of PCA-Compressed Data \n")
annotate!(X_pca[:, 1], X_pca[:, 2].-0.1, animal_names, annotationfontsize=8)

