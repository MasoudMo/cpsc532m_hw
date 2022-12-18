using Images, Plots

# Load image
I = load("dog.png")
(nRows,nCols) = size(I)

# Show image
plot(I)
gui()

# Convert to features
R = permutedims(channelview(I),[2,3,1])
X = reshape(float64.(R),(nRows*nCols,3))

# Convert from features back to image
R2 = reshape(X,(nRows,nCols,3))
I2 = colorview(RGB,permutedims(R2,[3,1,2]))


