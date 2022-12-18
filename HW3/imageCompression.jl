using Images, Plots
include("kMeans.jl")


function to_features(I)

    (nRows,nCols) = size(I)

    # Convert to features
    R = permutedims(channelview(I),[2,3,1])
    X = reshape(float64.(R),(nRows*nCols,3))

    return X
end


function to_image(X, nRows, nCols)

    # Convert from features back to image
    R = reshape(X,(nRows,nCols,3))
    I = colorview(RGB,permutedims(R,[3,1,2]))

    return I
end


function quantizeImage(img_name, b)

    # Load the image
    I = load(img_name)
    (nRows,nCols) = size(I)

    # Transform the image into features
    X = to_features(I)

    # Fit the kMeans model
    kmeans_model = kMeans(X, 2^b)

    return kmeans_model.y, kmeans_model.W, nRows, nCols
end


function deQuantizeImage(y, W, nRows, nCols)

    # Reconstructed features
    X = W[y, :]

    # From features to image
    I = to_image(X, nRows, nCols)

    return I
    
end