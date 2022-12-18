# Load data
using JLD
include("kMeansL1Error.jl")
X = load("clusterData2.jld","X")

# K-means clustering
k = 4
include("kMedians.jl")

min_error = 9999999999
chosen_y = 0
chosen_model = 0
for i in 1:50
    model = kMedians(X,k,doPlot=false)
    y = model.predict(X)
    error = kMeansL1Error(X, y, model.W)

    if error < min_error
        global min_error = error
        global chosen_y = y
        global chosen_model = model
        @printf("******** Error decresed to %f ********\n", error)
    end
end

include("clustering2Dplot.jl")
clustering2Dplot(X,chosen_y,chosen_model.W)
