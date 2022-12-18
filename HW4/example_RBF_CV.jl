using Printf
using Random

# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

include("leastSquares.jl")

# Data is sorted, so randomize the data first
n = size(X,1)
perm = randperm(n)
minErr = Inf
bestSigma = []

for sigma in 2.0.^(-15:15)
	validError = 0

	for fold_num in 1:10
		validStart = Int64(n/10 * (fold_num - 1) + 1)
		validEnd = Int64(n/10 * fold_num)

		validNdx = perm[validStart:validEnd] # Indices of validation examples
		trainNdx = perm[setdiff(1:n,validStart:validEnd)] # Indices of training examples
		Xtrain = X[trainNdx,:]
		ytrain = y[trainNdx]
		Xvalid = X[validNdx,:]
		yvalid = y[validNdx]

		# Train on the training set
		model_sigma = leastSquaresRBFL2(Xtrain,ytrain,sigma,10^(-12))

		# Compute the error on the validation set
		yhat_sigma = model_sigma.predict(Xvalid)
		validError += sum((yhat_sigma - yvalid).^2)/(n/2)
	end

	@printf("With sigma = %.3f, validError = %.2f\n",sigma,validError/10)

	# Keep track of the lowest validation error
	if validError < minErr
		global minErr = validError
		global bestSigma = sigma
	end
end

# Now fit the model based on the full dataset
model = leastSquaresRBFL2(X,y,bestSigma,10^(-12))

# Report the error on the test set
t = size(Xtest,1)
yhat = model.predict(Xtest)
testError = sum((yhat - ytest).^2)/t
@printf("With best sigma of %.3f, testError = %.2f\n",bestSigma,testError)

# Plot model
using Plots
scatter(X,y,legend=false,linestyle=:dot)
scatter!(Xtest,ytest,legend=false,linestyle=:dot)
Xhat = minimum(X):.1:maximum(X)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = model.predict(Xhat)
plot!(Xhat,yhat,legend=false)