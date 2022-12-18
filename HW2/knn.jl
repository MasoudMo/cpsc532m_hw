include("misc.jl") # Includes GenericModel typedef

function knn_predict(Xhat,X,y,k)
  (n,d) = size(X)
  (t,d) = size(Xhat)
  k = min(n,k) # To save you some debuggin

  # Compute the pairwise distance between all samples in the test and training set
  # The distancesSquared function computes the "squared" distance, which is OK since
  # we only care about relative distance differences
  distances = distancesSquared(Xhat, X)

  predictions = zeros(t)
  for i in 1:t
    sorted_idx = sortperm(distances[i, :])
    predictions[i] = mode(y[sorted_idx[1:k]])
  end

  return predictions
end

function knn(X,y,k)
	# Implementation of k-nearest neighbour classifier
  predict(Xhat) = knn_predict(Xhat,X,y,k)
  return GenericModel(predict)
end
