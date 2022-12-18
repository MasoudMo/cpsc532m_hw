include("misc.jl")
include("randomTree.jl")
include("decisionStump.jl")

function randomForest(X, y, depth, nTrees)
    subModels = Array{GenericModel}(undef,nTrees)
    
    for i in 1:nTrees
        subModels[i] = randomTree(X, y, depth)
    end

    function predict(Xhat)
        (t,d) = size(Xhat)
        yTrees = zeros(t, nTrees)
        yhat = zeros(t)
        
        for i in 1:nTrees
            yTrees[:, i] = subModels[i].predict(Xhat)
        end

        for i in 1:t
            yhat[i] = mode(yTrees[i, :])
        end

        return yhat

    end

    return GenericModel(predict)

end