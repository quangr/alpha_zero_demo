struct state
    scores1::Vector{Int}
    scores2::Vector{Int}
    actions::Vector{Int}
    done::Bool
    move::Int
    reward::Int
end
function printboard(s::UInt)
    s1 = s & 0xffff
    s2 = s >> 16
    for ind in 1:9
        i = fld(ind - 1, 3) + 1
        j = rem(ind - 1, 3) + 1
        if s1 & (1 << (ind - 1)) != 0
            print(" X ")
        elseif s2 & (1 << (ind - 1)) != 0
            print(" O ")
        else
            print("   ")
        end
        if j < 3
            print("|")
        else
            print("\n")
            if i < 3
                print("---|---|---\n")
            end
        end
    end
end
function treetovector(s::UInt)
    s1 = s & 0xffff
    s2 = s >> 16
    res = zeros(Int, 9)
    len = 0
    for i in 0:8
        if s1 & (1 << i) != 0
            res[i+1] = 1
            len += 1
        elseif s2 & (1 << i) != 0
            res[i+1] = -1
            len += 1
        end
    end
    return res, len
end
function state(s::UInt)
    scores1 = zeros(Int, 8)
    scores2 = zeros(Int, 8)
    actions = []
    s1 = s & 0xffff
    s2 = s >> 16
    done = false
    reward = 0
    for i in 0:8
        if s1 & (1 << i) != 0
            r, d = makemove!(scores1, i + 1, 1)
            if d
                done = true
                reward = r
            end
        elseif s2 & (1 << i) != 0
            r, d = makemove!(scores2, i + 1, 0)
            if d
                done = true
                reward = r
            end
        else
            push!(actions, i + 1)
        end
    end
    if s1 | s2 == 0x1ff
        done = true
    end
    move = 1 - (length(actions) % 2)
    state(scores1, scores2, actions, done, move, reward)
end

struct game
    statedict::Dict{Int,state}
end


InitState = convert(UInt, 0)
# InitState=convert(UInt,0x60011)

function statetype(a::game)
    return UInt
end

function actiontype(a::game)
    return Int
end

function makemove!(scores::Vector{Int}, a, move)
    r = 0
    done = false
    scores[a%3+1] += 1
    scores[4+div(a - 1, 3)] += 1
    if a % 4 == 1
        scores[7] += 1
    end
    if a % 2 == 1 && a >= 3 && a <= 7
        scores[8] += 1
    end
    if scores[4+div(a- 1, 3)] == 3 || scores[a%3+1] == 3 || scores[7] == 3 || scores[8] == 3
        done = true
    end
    if done
        r = move == 1 ? 1 : -1
    end
    return r, done
end

function step(mdp::game, s::UInt, a::Int)
    cachestate = get(mdp.statedict, s, state(s))
    done = false
    r = 0
    move = 1 - cachestate.move
    if move == 1
        scores = deepcopy(cachestate.scores1)
        s |= 1 << (a - 1)
    else
        scores = deepcopy(cachestate.scores2)
        s |= 1 << (a - 1 + 16)
    end
    r, done = makemove!(scores, a, move)
    if (s | (s >> 16)) & 0x1ff == 0x1ff &&!done
        done = true
        r = 0
    end
    if move == 1
        newstate = state(scores, cachestate.scores2, filter(x -> x != a, cachestate.actions), done, move, r)
    else
        newstate = state(cachestate.scores1, scores, filter(x -> x != a, cachestate.actions), done, move, r)
    end
    mdp.statedict[s] = newstate
    # printboard(s)
    # @show done
    # @show newstate.scores1
    # @show newstate.scores2
    return s, r, done
end



function isterminal(mdp::game, s::UInt)
    s = get(mdp.statedict, s, state(s))
    return s.done
end

function actions(mdp::game, s::UInt)
    s = get(mdp.statedict, s, state(s))
    return s.actions
end

function reward(mdp::game, s::UInt)
    s = get(mdp.statedict, s, state(s))
    return s.reward
end

function getreward(planner, s::UInt)
    s = get(planner.mdp.statedict, s, state(s))
    return s.reward
end


function getmove(mdp::game, s::UInt)
    s = get(mdp.statedict, s, state(s))
    return s.move
end

