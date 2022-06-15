ENV["GKSwstype"] = "100"
using Plots
gr()
xs = [string("x", i) for i = 1:3]
ys = [string("y", i) for i = 1:3]
z = [0.0, 0.0, 0.0875, 0.065, 0.0, 0.06, 0.1475, 0.145, 0.495]
z= reshape(z,3,3)
# heatmap(xs, ys, z, aspect_ratio = 1,c = cgrad([:white,:red]),axis=([], false))

hms = [heatmap(xs, ys, z, aspect_ratio = 1,c = cgrad([:white,:red]),axis=([], false)) for i in 1:2]
plot(hms..., layout = (1,2), colorbar = true,title=["洒几滴哦" "撒大苏打"],fontfamily="Source Han Serif CN")

str1 = ['x','o', 0.0875, 0.065, 'x', 0.06, 0.1475, 0.145, 0.495]
# str=reshape(str1,3,3)
# annotate!( vec(tuple.((1:length(xs))'.-0.5, (1:length(ys)).-0.5, string.(str))) ,subplot=1)
# str2 = ['x','o', 0.0875, 0.065, 'o', 0.06, 0.1475, 0.145, 0.495]
# str=reshape(str2,3,3)
# annotate!( vec(tuple.((1:length(xs))'.-0.5, (1:length(ys)).-0.5, string.(str))) ,subplot=2)


png("a")