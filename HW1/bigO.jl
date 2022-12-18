function func1(n)
	x = randn(n)
	biggest = -Inf
	for i = 1:n
		if x[i] > biggest
			biggest = x[i]
		end
	end
	return biggest
end

func2(n) = maximum(randn(n))

func3(n) = max(n,10)

function func4(n)
	X = zeros(n,n);
	for i in 1:n
		for j in 1:n
			X[i,j] = max(i,j)
		end
	end
	return X;
end

function func5(n)
	A = randn(n,n)
	B = randn(n,n)
	C = zeros(n,n)
	for i in 1:n
		for j in 1:n
			C[i,j] = dot(A[i,:],B[:,j])
		end
	end
	return C
end

	
