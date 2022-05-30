using MPI
using BenchmarkTools
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
@show size = MPI.Comm_size(comm)

N = 4
src=1
function asyncgetvalue(states,comm)
    statedata=Float32.(reshape(zeros(16*9),(9,16)))
    movedata=Int64.(zeros(16))
    for (i,state) in enumerate(states)
        statedata[:,i]=Float32.(state.state)
        movedata[i]=state.move
    end
    # @show senddata
    # rreq = MPI.Recv!(recv_mesg, src, 11, comm)
    MPI.Send(statedata, 1, 0, comm)
    MPI.Send(movedata, 1, 0, comm)
    recv_mesg = Array{Float32}(undef, (10,16))
    rreq = MPI.Recv!(recv_mesg, src, 0, comm)
    return recv_mesg'
    # print("$rank: Received $src -> $rank = $recv_mesg\n")
    # MPI.Send(Float64.(reshape(collect(1:4),(2,2))), 1, 1, comm)
end
# @btime getvalue(comm)
# MPI.Barrier(comm)

function softmax(x)
    exp.(x) ./ sum(exp.(x))
end

function reqvalue(p,s::Vector{Int},move)
    c=p.askch
    cs=p.recvch
    put!(c,message(threadid(),s,move))
    threadid()
    x=take!(cs[threadid()])
end

function getnetwork(planner,s::UInt)
    t,len=treetovector(s)
    # printboard(s)
    if (len%2)==planner.solver.move
        return reqvalue(planner.solver.policy,t,0)
        # println("now policy 0")
    else
        return reqvalue(planner.solver.policy,t,1)
    end
end
function getprior(planner,s::UInt)
    return softmax(getnetwork(planner,s)[2:10])
end
sigmoid(z) = 1.0 ./ (1.0 .+ exp.(-z))


function getvalue(planner,s::UInt)
    return 2*sigmoid(getnetwork(planner,s)[1])-1
end


function train(X,y)
    # function loss_fn(x, y)
    #     res=p.network(x)
    #     y=hcat(y...)
    #     mse=Flux.mse(2*Flux.sigmoid.(res[1,:]).-1, y[1,:])
    #     penalty=Flux.logitcrossentropy(res[2:10,:],y[2:10,:])
    #     println("mse:",mse)
    #     println("penalty:",penalty)
    #     mse+penalty
    # end
    # p=planner.solver.policy
    @show X[1:3]
    X=hcat(map(x->MCTS.treetovector(convert(UInt,x))[1],X)...)
    y=hcat(y...)
    MPI.Send(Float32.(X), 1, 1, comm)
    MPI.Send(Float32.(y), 1, 1, comm)
    # recv_mesg = Array{Float32}(undef, (10,16))
    # rreq = MPI.Recv!(recv_mesg, 1, 1, comm)
    # Flux.train!(loss_fn, Flux.params(p.network), [(X, y)], p.opt)
end
