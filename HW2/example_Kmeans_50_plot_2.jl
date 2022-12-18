# Load data
using JLD
include("kMeansL1Error.jl")
X = load("clusterData2.jld","X")

# K-means clustering
include("kMeans.jl")

min_errors = zeros(10)
for k in 1:10
    min_error = 9999999999
    for i in 1:50
        model = kMeans(X,k,doPlot=false)
        y = model.predict(X)
        error = kMeansL1Error(X, y, model.W)

        if error < min_error
            global min_errors[k] = error
            min_error = error
        end
    end
end

using Plots
plot(1:10, min_errors, title="Min kMeansL1Error For Different K")
xlabel!("K")
ylabel!("Min Error (over 50 runs)")


# include("clustering2Dplot.jl")
# clustering2Dplot(X,chosen_y,chosen_model.W)
