include("misc.jl")

function kMeansL1Error(X, y, W)
    (n,d) = size(X)

    error = 0

    for i in 1:n
        for j in 1:d
            error += abs(X[i, j] - W[y[i], j])
        end
    end

    return error
end