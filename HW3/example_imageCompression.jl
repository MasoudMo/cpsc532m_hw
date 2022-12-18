using Images, Plots
include("imageCompression.jl")

b = 6

y, W, nRows, nCols = quantizeImage("dog.png", b)

dequantized_image = deQuantizeImage(y, W, nRows, nCols)

# Show image
# plot(dequantized_image)
# gui()
save("./dog6b.png", dequantized_image)