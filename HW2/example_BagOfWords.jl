using JLD
using SparseArrays

data = load("newsgroups.jld")
X = data["X"]
y = data["y"]
Xtest = data["Xtest"]
ytest = data["ytest"]
wordlist = data["wordlist"]
groupnames = data["groupnames"]

# print("The 50th word in the wordlist is: \n")
# print(wordlist[50])
# print("\n \n")

# print("The following words are present in example 500: \n")
# for (idx, exists) in enumerate(convert(Array{Int}, X[500, :]))
#     if exists == 1
#         print(wordlist[idx])
#         print("\n")
#     end
# end

# print("\n")
# print("Example 500 comes from the following newsgroup: \n")
# print(groupnames[y[500]])
# print("\n \n \n")