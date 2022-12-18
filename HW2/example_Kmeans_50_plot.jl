# Load data
using JLD
include("kMeansError.jl")
X = load("clusterData.jld","X")

# K-means clustering
include("kMeans.jl")

min_errors = zeros(10)
for k in 1:10
    min_error = 9999999999
    for i in 1:50
        model = kMeans(X,k,doPlot=false)
        y = model.predict(X)
        error = kMeansError(X, y, model.W)

        if error < min_error
            global min_errors[k] = error
            min_error = error
        end
    end
end

using Plots
plot(1:10, min_errors, title="Min kMeansError For Different K")
xlabel!("K")
ylabel!("Min Error (over 50 runs)")


# include("clustering2Dplot.jl")
# clustering2Dplot(X,chosen_y,chosen_model.W)
