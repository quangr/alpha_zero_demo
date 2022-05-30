using Random
# function getvalue(s)
#     return rand()
# end

using Flux

function getpolicynn()
Chain(
    Dense(9, 64, Flux.relu),
    Dense(64, 64, Flux.relu),
    Dense(64, 64, Flux.relu),
    Dense(64, 10)
)
end

struct Policy
    network
    opt 
    function Policy(network)
        new(network,Flux.ADAM(0.01))
    end
end

function softmax(x)
    exp.(x) ./ sum(exp.(x))
end

function getnetwork(planner,s::UInt)
    t,len=treetovector(s)
    # printboard(s)
    if (len%2)==planner.solver.move
        network=planner.solver.policy.network
        # println("now policy 0")
    else
        network=planner.solver.oppolicy.network
        # println("now policy 1")
    end
    return network(t)
end
function getprior(planner,s::UInt)
    return softmax(getnetwork(planner,s)[2:10])
end
Sigmoid(z) = 1.0 ./ (1.0 .+ exp.(-z))


function getvalue(planner,s::UInt)
    return 2*Sigmoid(getnetwork(planner,s)[1])-1
end



# function getvalue(p::Policy,s::UInt)
#     return Sigmoid(p.network(treetovector(s))[1])
# end

function train(planner,X,y)
    function loss_fn(x, y)
        res=p.network(x)
        y=hcat(y...)
        mse=Flux.mse(2*Flux.sigmoid.(res[1,:]).-1, y[1,:])
        penalty=Flux.logitcrossentropy(res[2:10,:],y[2:10,:])
        println("mse:",mse)
        println("penalty:",penalty)
        mse+penalty
    end
    p=planner.solver.policy
    X=hcat(map(x->MCTS.treetovector(convert(UInt,x))[1],X)...)
    Flux.train!(loss_fn, Flux.params(p.network), [(X, y)], p.opt)
end


# loss_fn(x, y) = Flux.mse(model(x), y)
# opt = Flux.ADAM(0.1)
# N = 10000
# loss = zeros(Float32, N)
# for i in 1:N
#     Flux.train!(loss_fn, Flux.params(model), [(X, y)], opt)
#     loss[i] = loss_fn(X, y)
#     if i % 1000 == 0
#         println(loss[i])
#     end
# end