using DelimitedFiles
using Statistics
using Plots

include("misc.jl")

dataTable = readdlm("fluTrends.csv",',')

X = real(dataTable[2:end,:])

@show(minimum(X))

@show(maximum(X))

@show(mean(X))

@show(median(X))

@show(mode(X))

plot(X, x=:total_bill, kind="histogram")