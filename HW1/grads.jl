
### An example to get you started:

# Function
func0(x) = sum(x.^2)

# Gradient
grad0(x) = 2x

### Function 1
function func1(x)
	f = 0;
	n = length(x)
	for i in 1:n
		f += (1/2)*(x[i] - i)^2
	end
	return f
end

function grad1(x)
	n = length(x);
	g = zeros(n);
	for i in 1:n
		g[i] += x[i] - i
	end
	return g
end

### Function 2
using LinearAlgebra
func2(x) = dot(x,1:length(x))

function grad2(x)
	n = length(x);
	g = zeros(n);

	for i in 1:n
		g[i] = i
	end

	return g
end

### Function 3
func3(x) = sum(max.(x,0).^2)

function grad3(x)
	n = length(x);
	g = zeros(n);
	g[x .<= 0] .= 0
	g[x .> 0] = 2 * x[x .> 0]
	return g
end

### A function to approximate the derivative numerically
function numGrad(func,x)
	n = length(x);
	delta = 1e-6;
	g = zeros(n);
	fx = func(x);
	for i = 1:n
		e_i = zeros(n);
		e_i[i] = 1;
		g[i] = (func(x[:] + delta*e_i) - fx)/delta;
	end
	return g
end

