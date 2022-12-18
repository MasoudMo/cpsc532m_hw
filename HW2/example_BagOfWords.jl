using JLD
using SparseArrays

data = load("newsgroups.jld")
X = data["X"]
y = data["y"]
Xtest = data["Xtest"]
ytest = data["ytest"]
wordlist = data["wordlist"]
groupnames = data["groupnames"]