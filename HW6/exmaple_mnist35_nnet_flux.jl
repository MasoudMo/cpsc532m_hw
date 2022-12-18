# Load X and y variable
using JLD, Printf, Statistics
using Flux
using Flux.Optimise: update!
data = load("mnist35.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])
y[y.==2] .= 0
ytest[ytest.==2] .= 0
(n,d) = size(X)

feats_std = std(X)
feats_mean = mean(X)

# Standardize data
X = X.-feats_mean ./ feats_std
Xtest = Xtest.-feats_mean ./ feats_std

# Define a 3 hidden layer model using Flux
model = Chain(Dense(d, 30, tanh, init=Flux.kaiming_uniform), 
              Dense(30, 20, tanh, init=Flux.kaiming_uniform), 
              Dense(20, 5, tanh, init=Flux.kaiming_uniform),
              Dense(5, 1, sigmoid, init=Flux.kaiming_uniform))
              
# Define the loss function
loss(x, y) = Flux.binarycrossentropy(model(x), y)

# Extract model params to be used for grad computation
ps = params(model)

# Define the optimizer
stepSize = 1e-2
opt = Descent(stepSize)

# Train with mini batches
maxIter = 50000
samples_per_batch = 361
patience = 3
prev_err = 9999
for t in 1:maxIter

	i = rand(1:n, samples_per_batch)

	grads = Flux.gradient(ps) do 
		loss(transpose(X[i, :]), reshape(y[i], 1, size(y[i], 1)))
	end
	update!(opt, ps, grads)

	# Every few iterations, plot the data/model:
	if (mod(t-1,round(1000)) == 0)
		yhat = model(transpose(Xtest)) .> 0.5
		err = sum(yhat .!= reshape(ytest, 1, size(ytest, 1)))/size(Xtest,1)
		@printf("Training iteration = %d, error rate = %.2f\n",t-1,err)

		if err < prev_err
            global patience = 3
            global prev_err = err
        else
            if patience == 1
                global stepSize /= 10
				global opt = Descent(stepSize)
                @printf("Reduced step size to %.4f", stepSize)
                global patience = 3
            else
                global patience -= 1
            end
        end
	end
end
