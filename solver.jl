using Random
using StatsBase

mutable struct MCTSTree{S,A}
    state_map::Dict{S,Int}

    # these vectors have one entry for each state node
    child_ids::Vector{Vector{Int}}
    total_n::Vector{Int}
    s_labels::Vector{S}

    # these vectors have one entry for each action node
    n::Vector{Int}
    q::Vector{Float64}
    a_labels::Vector{A}
    p::Vector{Float64}

    _vis_stats::Union{Nothing, Dict{Pair{Int,Int}, Int}} # maps (said=>sid)=>number of transitions. THIS MAY CHANGE IN THE FUTURE

    function MCTSTree{S,A}(sz::Int=1000) where {S,A}
        sz = min(sz, 100_000)

        return new(Dict{S, Int}(),

                   sizehint!(Vector{Int}[], sz),
                   sizehint!(Int[], sz),
                   sizehint!(S[], sz),

                   sizehint!(Int[], sz),
                   sizehint!(Float64[], sz),
                   sizehint!(A[], sz),
                   sizehint!(Float64[], sz),
                   Dict{Pair{Int,Int},Int}()
                  )
    end
end
struct StateNode{S,A}
    tree::MCTSTree{S,A}
    id::Int
end
@inline state(n::StateNode) = n.tree.s_labels[n.id]
@inline total_n(n::StateNode) = n.tree.total_n[n.id]
@inline child_ids(n::StateNode) = n.tree.child_ids[n.id]
@inline children(n::StateNode) = (ActionNode(n.tree, id) for id in child_ids(n))


mutable struct MCTSSolver 
    n_iterations::Int64
    max_time::Float64
    depth::Int64
    exploration_constant::Float64
    rng::AbstractRNG
    # estimate_value::Any
    init_Q::Any
    init_N::Any
    reuse_tree::Bool
    enable_tree_vis::Bool
    timer::Function
    policy::Policy
    move::Int
end

function MCTSSolver(;n_iterations::Int64=100,
    max_time::Float64=Inf,
    depth::Int64=10,
    exploration_constant::Float64=1.0,
    rng=Random.GLOBAL_RNG,
    # estimate_value=RolloutEstimator(RandomSolver(rng)),
    init_Q=0.0,
    init_N=0,
    reuse_tree::Bool=false,
    enable_tree_vis::Bool=false,
    timer=() -> 1e-9 * time_ns(),
    policy=nothing,
    move=0
    )
return MCTSSolver(n_iterations, max_time, depth, exploration_constant, rng, #estimate_value, 
init_Q, init_N,
     reuse_tree, enable_tree_vis, timer,policy,move)
end

function plan!(planner, s)
    tree = build_tree(planner, s)
    planner.tree = tree
    return tree
end

function best_sanode_Q(snode::StateNode)
    best_Q = -Inf
    best = first(children(snode))
    # print(collect((q(sanode),action(sanode)) for sanode in children(snode)))
    for sanode in children(snode)
        if q(sanode) > best_Q
            best_Q = q(sanode)
            best = sanode
        end
    end
    return best
end




solve(solver::MCTSSolver, mdp) = MCTSPlanner(solver, mdp)

Base.isempty(t::MCTSTree) = isempty(t.state_map)
state_nodes(t::MCTSTree) = (StateNode(t, id) for id in 1:length(t.total_n))

StateNode(tree::MCTSTree{S}, s::S) where S = StateNode(tree, tree.state_map[s])

mutable struct MCTSPlanner{P, S, A,  RNG}
	solver::MCTSSolver # containts the solver parameters
	mdp::P # model
    tree::Union{Nothing,MCTSTree{S,A}} # the search tree
    # solved_estimate::SE
    rng::RNG
end

function MCTSPlanner(solver::MCTSSolver, mdp)
    # tree = Dict{statetype(mdp), StateNode{actiontype(mdp)}}()
    tree = MCTSTree{statetype(mdp), actiontype(mdp)}(solver.n_iterations)
    # se = convert_estimator(solver.estimate_value, solver, mdp)
    # return MCTSPlanner(solver, mdp, tree, se, solver.rng)
    return MCTSPlanner(solver, mdp, tree, solver.rng)
end

struct ActionNode{S,A}
    tree::MCTSTree{S,A}
    id::Int
end


function action_info(p, s)
    tree = plan!(p, s)
    probs=zeros(9)
    for sanode in children(StateNode(tree, s))
        probs[action(sanode)]=n(sanode)
    end
    # @show probs
    probs=probs/sum(probs)
    # best = best_sanode_Q(StateNode(tree, s))
    return sample(Weights(probs)), (tree=tree,),probs
end



@inline action(n::ActionNode) = n.tree.a_labels[n.id]
@inline n(n::ActionNode) = n.tree.n[n.id]
@inline q(n::ActionNode) = n.tree.q[n.id]
@inline p(n::ActionNode) = n.tree.p[n.id]


action(p, s) = first(action_info(p, s))

function simulate(planner, node::StateNode, depth::Int64)
    mdp = planner.mdp
    rng = planner.rng
    s = state(node)
    tree = node.tree
    # once depth is zero return
    if isterminal(planner.mdp, s)||depth == 0
	return getvalue(planner,s)
    end

    # pick action using UCT
    sanode = best_sanode_UCB(node, planner.solver.exploration_constant)
    said = sanode.id

    # transition to a new state
    sp, r,done = step(mdp, s, action(sanode))
    spid = get(tree.state_map, sp, 0)
    if spid == 0
        spn = insert_node!(tree, planner, sp)
        spid = spn.id
        q = r + getvalue(planner,s)
    else
        q = r + simulate(planner, StateNode(tree, spid) , depth-1)
    end
    # if planner.solver.enable_tree_vis
    #     record_visit!(tree, said, spid)
    # end

    tree.total_n[node.id] += 1
    tree.n[said] += 1
    tree.q[said] += (q - tree.q[said]) / tree.n[said] # moving average of Q value
    return q
end



function build_tree(planner, s)
    n_iterations = planner.solver.n_iterations
    depth = planner.solver.depth

    if planner.solver.reuse_tree
        tree = planner.tree
    else
        tree = MCTSTree{statetype(planner.mdp), actiontype(planner.mdp)}(n_iterations)
    end

    sid = get(tree.state_map, s, 0)
    if sid == 0
        root = insert_node!(tree, planner, s)
    else
        root = StateNode(tree, sid)
    end

    timer = planner.solver.timer
    start_s = timer()
    # build the tree
    for n = 1:n_iterations
        simulate(planner, root, depth)
        if timer() - start_s >= planner.solver.max_time
            break
        end
    end
    return tree
end
function init_N end
init_N(f::Function, mdp, s, a) = f(mdp, s, a)
init_N(n::Number, mdp, s, a) = convert(Int, n)

function init_Q end
init_Q(f::Function, mdp, s, a) = f(mdp, s, a)
init_Q(n::Number, mdp, s, a) = convert(Float64, n)

function insert_node!(tree::MCTSTree, planner::MCTSPlanner, s)
    push!(tree.s_labels, s)
    tree.state_map[s] = length(tree.s_labels)
    push!(tree.child_ids, [])
    total_n = 0
    @assert (s>>16)&s==0
    prior=getprior(planner,s)
    acts=actions(planner.mdp, s)
    for a in acts
        n = init_N(planner.solver.init_N, planner.mdp, s, a)
        total_n += n
        push!(tree.n, n)
        push!(tree.q, init_Q(planner.solver.init_Q, planner.mdp, s, a))
        push!(tree.p,prior[a])
        push!(tree.a_labels, a)
        push!(last(tree.child_ids), length(tree.n))
    end
    push!(tree.total_n, total_n)
    return StateNode(tree, length(tree.total_n))
end

function best_sanode_UCB(snode::StateNode, c::Float64)
    best_UCB = -Inf
    best = first(children(snode))
    sn = total_n(snode)
    for sanode in children(snode)
        
        UCB = q(sanode) + c*p(sanode)*sqrt(sn)/(1+n(sanode))		
        if isnan(UCB)
            @show sn
            @show n(sanode)
            @show q(sanode)
        end
		
        @assert !isnan(UCB)
        @assert !isequal(UCB, -Inf)
		
        if UCB > best_UCB
            best_UCB = UCB
            best = sanode
        end
    end
    return best
end
