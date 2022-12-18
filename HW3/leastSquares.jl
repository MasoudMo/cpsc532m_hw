include("misc.jl")

function leastSquares(X,y)

	# Find regression weights minimizing squared error
	w = (X'X)\(X'y)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end


function leastSquaresBias(X,y)

	(n, d) = size(X)

	# Add a 1's column to add bias
	Z = hcat(ones(n, 1), X)

	# Find regression weights minimizing squared error
	v = (Z'Z)\(Z'y)

	# Make linear prediction function
	function predict(Xhat)
		nhat = size(Xhat)[1]
		Xhat = hcat(ones(nhat, 1), Xhat)
		return Xhat*v
	end

	# Return model
	return GenericModel(predict)
end


function leastSquaresBasis(X, y, p)

	# Make the new basis
	Z = polyBasis(X, p)

	# Find regression weights minimizing squared error
	v = (Z'Z)\(Z'y)

	# Make linear prediction function
	function predict(Xhat)
		Xhat = polyBasis(Xhat, p)
		return Xhat*v
	end

	# Return model
	return GenericModel(predict)
end


function polyBasis(X, p)
	n = size(X)[1]

	Z = zeros(n, p+1)

	for i in 0:p
		Z[:, i+1] = X.^i
	end

	return Z
end


function leastSquaresRBF(X, y, sigma)

	# Make the new basis
	Z = gausRBFBasis(X, X, sigma)

	# Find regression weights minimizing squared error
	v = (Z'Z)\(Z'y)

	# Make linear prediction function
	function predict(Xhat)
		Xhat = gausRBFBasis(Xhat, X, sigma)
		return Xhat*v
	end

	# Return model
	return GenericModel(predict)
end


function gausRBFBasis(X1, X2, sigma)
    
    # Create the matrix containing L2 distance squared
    D = distancesSquared(X1, X2);

    # Compute the basis
    Z = exp.(-1 .* D / (2*sigma^2));

    return Z
end


function weightedLeastSquares(X,y,v)

	# Diagnolize v
	V = Diagonal(v)
	
	# Find regression weights minimizing squared error
	w = (X'*V*X) \ (X'*V*y)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end