using Flux
X = Float32[
    0.0 0 1 1; 
    0.0 1 0 1
    ]
y = Float32[0.0 1 1 0]

model = Chain(
    Dense(9, 30, Flux.sigmoid),
    Dense(30, 10)
)

struct policy
    network
end
function getprior(p::policy,s::Vector{Int})
    return p.network(s)[2:10]
end

function getvalue(p::policy,s::Vector{Int})
    return Sigmoid(model(collect(1:9))[1])
end

function getvalue(p::policy,s::UInt)
    return Sigmoid(model(collect(1:9))[1])
end


loss_fn(x, y) = Flux.mse(model(x), y)
opt = Flux.ADAM(0.1)
N = 10000
loss = zeros(Float32, N)
for i in 1:N
    Flux.train!(loss_fn, Flux.params(model), [(X, y)], opt)
    loss[i] = loss_fn(X, y)
    if i % 1000 == 0
        println(loss[i])
    end
end