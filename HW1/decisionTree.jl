include("decisionStump.jl")

function decisionTree(X,y,depth)
	# Fits a decision tree using greedy recursive splitting
	# (uses recursion to make the code simpler)

	(n,d) = size(X)

	# Learn a decision stump
	splitModel = decisionStump(X,y)

	if depth <= 1 || splitModel.baseSplit
		# Base cases where we stop splitting:
		# - this stump gets us to the max depth
		# - this stump doesn't split the data
		return splitModel
	else
		# Use the decision stump to split the data
		yes = splitModel.split(X)

		# Recusively fit a decision tree to each split
		yesModel = decisionTree(X[yes,:],y[yes],depth-1)
		noModel = decisionTree(X[.!yes,:],y[.!yes],depth-1)
		
		# Make a predict function
		function predict(Xhat)
			(t,d) = size(Xhat)
			yhat = zeros(t)

			yes = splitModel.split(Xhat)

			yhat[yes] = yesModel.predict(Xhat[yes,:])
			yhat[.!yes] = noModel.predict(Xhat[.!yes,:])
			return yhat
		end

		# function predict(Xhat)
		# 	(t,d) = size(Xhat)
		# 	yhat = zeros(t)

		# 	for sample_idx in 1:t
		# 		if Xhat[sample_idx, 2] >= 37.695206
		# 			yhat[sample_idx] = 1
		# 		else
		# 			if Xhat[sample_idx, 1] >= -96.032692
		# 				yhat[sample_idx] = 1
		# 			else
		# 				if Xhat[sample_idx, 1] >= -112.548331
		# 					yhat[sample_idx] = 2
		# 				else
		# 					yhat[sample_idx] = 1
		# 				end
		# 			end
		# 		end
		# 	end

		# 	return yhat
		# end

		return GenericModel(predict)
	end
end

	
function decisionTreeHardCoded(X,y,depth)
	# Fits a decision tree using greedy recursive splitting
	# (uses recursion to make the code simpler)

	(n,d) = size(X)

	# Learn a decision stump
	splitModel = decisionStump(X,y)

	if depth <= 1 || splitModel.baseSplit
		# Base cases where we stop splitting:
		# - this stump gets us to the max depth
		# - this stump doesn't split the data
		return splitModel
	else
		# Use the decision stump to split the data
		yes = splitModel.split(X)

		# Recusively fit a decision tree to each split
		yesModel = decisionTree(X[yes,:],y[yes],depth-1)
		noModel = decisionTree(X[.!yes,:],y[.!yes],depth-1)

		function predict(Xhat)
			(t,d) = size(Xhat)
			yhat = zeros(t)

			for sample_idx in 1:t
				if Xhat[sample_idx, 2] >= 37.695206
					yhat[sample_idx] = 1
				else
					if Xhat[sample_idx, 1] >= -96.032692
						yhat[sample_idx] = 1
					else
						if Xhat[sample_idx, 1] >= -112.548331
							yhat[sample_idx] = 2
						else
							yhat[sample_idx] = 1
						end
					end
				end
			end

			return yhat
		end

		return GenericModel(predict)
	end
end