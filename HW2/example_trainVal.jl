using Printf

# Load X and y variable
using JLD
X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
n = size(X,1)

Xval = X[1:n÷2, :]
yval = y[1:n÷2]
Xtrain = X[n÷2:n, :]
ytrain = y[n÷2:n]

trainErrors = zeros(15)
valErrors = zeros(15)
depths = 1:15

# Train a depth-2 decision tree
for depth in depths
    include("decisionTree_infoGain.jl")
    model = decisionTree_infoGain(Xtrain,ytrain,depth)

    # Evaluate the training error
    yhat = model.predict(Xtrain)
    trainError = sum(yhat .!= ytrain)/(n/2)
    @printf("Train error with depth-%d decision tree: %.3f\n",depth,trainError)
    trainErrors[depth] = trainError

    # Evaluate the test error
    yhat = model.predict(Xval)
    valError = sum(yhat .!= yval)/(n/2)
    @printf("Validation error with depth-%d decision tree: %.3f\n",depth,valError)
    valErrors[depth] = valError
end  

using Plots
plot(depths, hcat(trainErrors, valErrors), title="Error vs. Depth", label=["Training" "Validation"])
xlabel!("Depth")
ylabel!("Error")

