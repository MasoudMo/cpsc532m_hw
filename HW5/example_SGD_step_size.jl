using Printf
using Statistics
include("misc.jl")

# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit least squares with bias via gradient descent
n = size(X,1)
Z = [ones(n,1) X]
lambda = 1

############################# Using gamma/t for step size ###########################

for power in -10:0

    gamma = 10.0^power

    global v = zeros(2,1)

    for t in 1:500

        local alpha = gamma / t
        
        # Randomly choose i
        i = rand(1:n)

        global v -= alpha*((v'*(Z[i, :]) - y[i, :]).*Z[i, :] + (lambda/n)*v)

    end

    predict(Xhat) = [ones(size(Xhat,1),1) Xhat]*v
    local model = LinearModel(predict,v)

    # Evaluate training error
    local yhat = model.predict(X)
    @printf("gamma/t with power = %f Squared train Error with least squares: %.3f\n",power, mean((yhat - y).^2))

    # Evaluate test error
    local yhat = model.predict(Xtest)
    @printf("gamma/t power = %f Squared test Error with least squares: %.3f\n",power, mean((yhat - ytest).^2))

end

############################# Using gamma/sqrt(t) for step size ###########################

for power in -10:0

    gamma = 10.0^power

    global v = zeros(2,1)

    for t in 1:500

        local alpha = gamma / sqrt(t)
        
        # Randomly choose i
        i = rand(1:n)

        global v -= alpha*((v'*(Z[i, :]) - y[i, :]).*Z[i, :] + (lambda/n)*v)

    end

    predict(Xhat) = [ones(size(Xhat,1),1) Xhat]*v
    local model = LinearModel(predict,v)

    # Evaluate training error
    local yhat = model.predict(X)
    @printf("gamma/sqrt(t) with power = %f Squared train Error with least squares: %.3f\n",power, mean((yhat - y).^2))

    # Evaluate test error
    local yhat = model.predict(Xtest)
    @printf("gamma/sqrt(t) power = %f Squared test Error with least squares: %.3f\n",power, mean((yhat - ytest).^2))

end

############################# Using gamma for step size ###########################

for power in -10:0

    gamma = 10.0^power

    global v = zeros(2,1)

    for t in 1:500

        local alpha = gamma
        
        # Randomly choose i
        i = rand(1:n)

        global v -= alpha*((v'*(Z[i, :]) - y[i, :]).*Z[i, :] + (lambda/n)*v)

    end

    predict(Xhat) = [ones(size(Xhat,1),1) Xhat]*v
    local model = LinearModel(predict,v)

    # Evaluate training error
    local yhat = model.predict(X)
    @printf("gamma with power = %f Squared train Error with least squares: %.3f\n",power, mean((yhat - y).^2))

    # Evaluate test error
    local yhat = model.predict(Xtest)
    @printf("gamma power = %f Squared test Error with least squares: %.3f\n",power, mean((yhat - ytest).^2))

end