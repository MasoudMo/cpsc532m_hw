using Printf
using Statistics

# Load X and y variable
using JLD
dataName = "citiesSmall.jld"
X = load(dataName,"X")
y = load(dataName,"y")
Xtest = load(dataName,"Xtest")
ytest = load(dataName,"ytest")

# Fit a KNN classifier
k = 1
include("knn.jl")
model = knn(X,y,k)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with %d-nearest neighbours: %.3f\n",k,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with %d-nearest neighbours: %.3f\n",k,testError)

# Evaluate the coordinates of Vancouver
Xvan = zeros(1, 2)
Xvan[1, 1] = 49.2827
Xvan[1, 2] = -123.1207
yhat = model.predict(Xvan)
print("Vancouver's label: ")
print(yhat)
print("\n")

include("plot2Dclassifier.jl")
plot2Dclassifier(X, y, model)