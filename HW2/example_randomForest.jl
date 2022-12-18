using Printf
using Statistics

# Load data
using JLD
fileName = "vowel.jld"
X = load(fileName,"X")
y = load(fileName,"y")
Xtest = load(fileName,"Xtest")
ytest = load(fileName,"ytest")

# Fit a random forest classifier
include("randomForest.jl")
depth = Inf
nTrees = 50
model = randomForest(X,y,depth,nTrees)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with depth-%d random forest: %.3f\n",depth,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with depth-%d random forest: %.3f\n",depth,testError)