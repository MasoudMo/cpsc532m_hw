using Printf
using Statistics

# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit a least squares model
include("leastSquares.jl")

sigma = 10

model = leastSquaresRBF(X,y, sigma)

# Evaluate training error
yhat = model.predict(X)
trainError = mean((yhat - y).^2)
@printf("sigma=%f ---- Squared train Error with least squares: %.3f\n",sigma, trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean((yhat - ytest).^2)
@printf("sigma=%f ---- Squared test Error with least squares: %.3f\n", sigma, testError)

@printf("\n")

# Plot model
using Plots
scatter(X,y,legend=false,linestyle=:dot)
Xhat = minimum(X):.1:maximum(X)
yhat = model.predict(reshape(Xhat, size(Xhat)[1], 1))
plot!(Xhat,yhat,legend=false)
gui()