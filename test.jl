using Base.Threads
ENV["GKSwstype"] = "100"
using Plots
gr()
using StatsBase
mutable struct playdata
    s::UInt
    p::Vector{Float64}
    gamewinner::Int
    # index::Int
end
# using Debugger
# break_on(:error)
include("mcts.jl")
using .MCTS
mdp = game(Dict())
function runonce(planner, InitState)
    done = false
    s = InitState
    playdatas = []
    planner.solver.move=0
    planner.solver.exploration_constant=.2
    while !done
        # println("nowstate:")
        # printboard(s)
        a, _, prob = action_info(planner, s)
        # @show prob
        state, reward, d = MCTS.step(planner.mdp, s, a)
        push!(playdatas, playdata(s, prob, planner.solver.move == 0 ? 1 : -1))
        if d
            # @show reward
            for data in playdatas
                data.gamewinner *= reward
            end
            # printboard(state)
        end
        planner.solver.move = 1 - planner.solver.move
        planner.solver.exploration_constant*=0.95
        done = d
        s = state
    end
    return playdatas
end


function testonce(planner, InitState)
    done = false
    s = InitState
    reward=0
    planner.solver.move=1
    while !done
        # println("nowstate:")
        # printboard(s)
        a, _, prob = action_info(planner, s)
        @show prob
        state, reward, d = MCTS.step(planner.mdp, s, a)
        if d
            return reward
            # printboard(state)
        end
        planner.solver.move = 1 - planner.solver.move
        done = d
        s = state
    end
    return reward
end

function winrate(planner,time=100)
    reward=[]
    for i in 1:time
        append!(reward, testonce(planner, InitState))
    end
    println("winrate:",length(filter(x->x!=-1,reward))/time)
end


function run_train(policy)
    for _ in 1:400
        # @time winrate(planner)
        function taskone(ch::Channel)
            @threads for i in 1:32
                # @show threadid()
                solver = MCTSSolver(n_iterations=400, depth=9, exploration_constant=.2,reuse_tree=false,policy=policy)
                planner = solve(solver, mdp)
                put!(ch,runonce(planner, InitState))
            end
        end
        chnl=Channel(taskone)
        X = []
        y = []
        for playdatas in chnl
            for playdata in playdatas
                push!(X, playdata.s)
                push!(y, vcat(convert(Float32,playdata.gamewinner), playdata.p))
            end
        end
        for i in 1:3
            index=sample(1:length(y),512)
            train(X[index],y[index])
        end
    end
end
policy=Policy()
@spawn listen(policy)
# run_train(policy)

function vectortouint(x::Vector{Int})
    output=UInt(0)
    len=0
    for i in 1:9
        if x[i]==1
            output|=1<<(i-1)
            len+=1
        elseif x[i]==-1
            output|=1<<(i-1+16)
            len+=1
        end
    end
    return output,len
end
function savefig(s::UInt,prob,value)
    prob=round.(prob; digits=3)
    value=round.(value; digits=3)
    xs = [string("x", i) for i = 1:3]
    ys = [string("y", i) for i = 1:3]
    hms = [heatmap(xs, ys, reshape(prob,3,3), aspect_ratio = 1,c = cgrad([:white,:red]),axis=([], false)),heatmap(xs, ys, reshape(value,3,3), aspect_ratio = 1,clim=(-1.,1.),c = cgrad([:blue,:white,:red]),axis=([], false))]
    plot(hms..., layout = (1,2), colorbar = true,title = ["P for next action"  "estimate for Q value"],)
    state=MCTS.treetovector(s)[1]
    str1=[]
    str2=[]
    for i in 1:9
        if state[i]==1
            push!(str1,'x')
            push!(str2,'x')
        elseif state[i]==-1
            push!(str1,'o')
            push!(str2,'o')
        else
            push!(str1,string(prob[i]))
            push!(str2,string(value[i]))
        end
    end
    annotate!( vec(tuple.((1:length(xs))'.-0.5, (1:length(ys)).-0.5, string.(reshape(str1,3,3)))) ,subplot=1)
    annotate!( vec(tuple.((1:length(xs))'.-0.5, (1:length(ys)).-0.5, string.(reshape(str2,3,3)))) ,subplot=2)
    # str2 = ['x','o', 0.0875, 0.065, 'o', 0.06, 0.1475, 0.145, 0.495]
    # str=reshape(str2,3,3)
    # annotate!( vec(tuple.((1:length(xs))'.-0.5, (1:length(ys)).-0.5, string.(str))) ,subplot=2)
    png("img/2/"*string(s))
end
function runonce(planner, InitState::Vector{Int})
    done = false
    s,len = vectortouint(InitState)
    playdatas = []
    planner.solver.move=len%2
    planner.solver.exploration_constant=.2*(.95)^len
    while !done
        println("nowstate:")
        printboard(s)
        a, tree, prob = action_info(planner, s)
        # @show prob
        savefig(s,prob,tree.Q)
        state, reward, d = MCTS.step(planner.mdp, s, a)
        push!(playdatas, playdata(s, prob, planner.solver.move == 0 ? 1 : -1))
        if d
            # @show reward
            for data in playdatas
                data.gamewinner *= reward
            end
            printboard(state)
        end
        planner.solver.move = 1 - planner.solver.move
        planner.solver.exploration_constant*=0.95
        done = d
        s = state
    end
    return playdatas
end


solver = MCTSSolver(n_iterations=20, depth=9, exploration_constant=5.,reuse_tree=false,policy=policy,debug=false)
planner = solve(solver, mdp)
runonce(planner,[1,0,1,0,-1,0,0,0,0])
