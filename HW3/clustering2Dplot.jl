using Plots

function clustering2Dplot(X,y,W=[])
	k = length(unique(y))

	markers = [:circle
		:rect
		:star5
		:diamond
		:hexagon
		:cross
		:xcross
		:utriangle
		:dtriangle
		:rtriangle
		:ltriangle
		:pentagon
		:octagon
		:star4
		:star6
		:star7
		:star8
		:vline
		:hline
		:+
		:-]

	for c in 1:k
		if c == 1
			scatter(X[y.==1,1],X[y.==1,2],legend=false,markershape=markers[1],color=c)
		else
			scatter!(X[y.==c,1],X[y.==c,2],legend=false,markershape=markers[c],color=c)
		end

		if !isempty(W)
			scatter!([W[c,1]],[W[c,2]],legend=false,markershape=markers[c],color=c,markersize=12)
		end

	end

	if any(y.==0)
		scatter!(X[y.==0,1],X[y.==0,2],legend=false,markershape=:circle,markersize=2,color=:black)
	end

	scatter!()
end

