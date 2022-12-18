# Load data
using JLD
X = load("clusterData2.jld","X")

# Density-based Clustering
radius = 20
minPts = 2
include("dbCluster.jl")
y = dbCluster(X,radius,minPts,doPlot=true)

include("clustering2Dplot.jl")
clustering2Dplot(X,y)

# Restrict plot to non-outlier range if wanted
#scatter!(xlims=[-20,20],ylims=[-15,25]) 