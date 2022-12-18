include("misc.jl")

function kMeansError(X, y, W)
    (n,d) = size(X)

    error = 0

    for i in 1:n
        for j in 1:d
            error += (X[i, j] - W[y[i], j])^2
        end
    end

    return error
end