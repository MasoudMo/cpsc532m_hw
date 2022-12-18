using Printf
using Statistics

# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit a least squares model
include("leastSquares.jl")

# Initialize loss variables
max_p = 10
trainErrors = zeros(max_p+1)
testErrors = zeros(max_p+1)

for p in 0:max_p
    model = leastSquaresBasis(X,y, p)

    # Evaluate training error
    yhat = model.predict(X)
    trainError = mean((yhat - y).^2)
    trainErrors[p+1] = trainError
    @printf("P=%d ---- Squared train Error with least squares: %.3f\n",p, trainError)

    # Evaluate test error
    yhat = model.predict(Xtest)
    testError = mean((yhat - ytest).^2)
    testErrors[p+1] = testError
    @printf("P=%d ---- Squared test Error with least squares: %.3f\n",p, testError)

    @printf("\n")

    # Plot model
    using Plots
    scatter(X,y,legend=false,linestyle=:dot)
    Xhat = minimum(X):.1:maximum(X)
    yhat = model.predict(Xhat)
    plot!(Xhat,yhat,legend=false)
    gui()
end

using Plots
plot(0:max_p, hcat(trainErrors, testErrors), title="Error vs. P", label=["Training" "Test"])
xlabel!("Polynomial Degree")
ylabel!("Error")