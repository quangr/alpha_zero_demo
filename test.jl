using Base.Threads
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
    while !done
        # println("nowstate:")
        # printboard(s)
        a, _, prob = action_info(planner, s)
        # print(prob)
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
        done = d
        s = state
    end
    return playdatas
end


function testonce(planner, InitState)
    done = false
    s = InitState
    reward=0
    planner.solver.move=0
    while !done
        # println("nowstate:")
        # printboard(s)
        a, _, prob = action_info(planner, s)
        # print(prob)
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
                solver = MCTSSolver(n_iterations=50, depth=5, exploration_constant=5.0,reuse_tree=false,policy=policy)
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
        for i in 1:5
            index=sample(1:length(y),512)
            train(X[index],y[index])
        end
    end
end
policy=Policy()
@spawn listen(policy)
run_train(policy)
