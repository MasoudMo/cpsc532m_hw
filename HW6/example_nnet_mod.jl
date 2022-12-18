using Printf
include("findMin.jl")
# Load X and y variable
using JLD
using Plots
data = load("basisData.jld")
(X,y) = (data["X"],data["y"])
(n,d) = size(X)

# Add the bias term
X = hcat(ones(n, 1), X)
(n,d) = size(X)

# Choose network structure and randomly initialize weights
include("NeuralNet.jl")
nHidden = [15, 5, 3]
nParams = NeuralNet_nParams(d,nHidden)
w = randn(nParams,1)

# Train with stochastic gradient
maxIter = 100000
# stepSize = 1e-4
# for t in 1:maxIter

# 	# The stochastic gradient update:
# 	i = rand(1:n, 200)
# 	(f,g) = NeuralNet_backprop_mod(w,X[i,:],y[i],nHidden)
# 	global w = w - stepSize*g

# 	# Every few iterations, plot the data/model:
# 	if (mod(t-1,round(maxIter/50)) == 0)
# 		@printf("Training iteration = %d\n",t-1)
# 		xVals = -10:.05:10
# 		Xhat = zeros(length(xVals),1)
# 		Xhat[:] .= xVals
# 		Xhat = hcat(ones(length(xVals), 1), Xhat)
# 		yhat = NeuralNet_predict(w,Xhat,nHidden)
# 		scatter(X[:, 2],y,legend=false,linestyle=:dot)
# 		plot!(Xhat[:, 2],yhat,legend=false)
# 		gui()
# 		sleep(.1)
# 	end
# end

objective(w) = NeuralNet_backprop_mod(w,X,y,nHidden)
w = findMin(objective,w,maxIter=maxIter)

xVals = -10:.05:10
Xhat = zeros(length(xVals),1)
Xhat[:] .= xVals
Xhat = hcat(ones(length(xVals), 1), Xhat)
yhat = NeuralNet_predict(w,Xhat,nHidden)
scatter(X[:, 2],y,legend=false,linestyle=:dot)
plot!(Xhat[:, 2],yhat,legend=false)
gui()

plot!()
