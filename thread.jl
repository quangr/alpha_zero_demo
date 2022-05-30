include("mpitest.jl")
using Base.Threads
struct Policy
    askch::Channel
    recvch::Vector{Channel}
    function Policy()
        recvch=collect((Channel() for i in 1:nthreads()))
        askch=Channel(16)
        new(askch,recvch)
    end
end
struct message
index::Int
state
move::Int
end
## ------------------------------------------------------------
# Produces input values
function producer(p::Policy)
    c=p.askch
    cs=p.recvch
    while true
        # println("Generating | ", n, " at: ", threadid())
        put!(c,message(threadid(),rand(1:10).*ones(9),1))
        threadid()
        x=take!(cs[threadid()])
        # @show x
        sleep(rand())
    end
end

## ------------------------------------------------------------
# Expensive function
function f(n::Int64)
    println("Running | ", n, " at: ", threadid())
    sleep(2.0)
end

## ------------------------------------------------------------
# Or in julia > v1.6
function listen(p::Policy)
    println("start listening")
    Ch=p.askch
    cs=p.recvch
    while true
        states=[]
        push!(states,take!(Ch))
        sleep(0.0001)
        while isready(Ch)
            push!(states,take!(Ch))
        end
        # print(collect((state.n.*ones(9) for state in states)))
        value=asyncgetvalue(states,comm)
        for (i,state) in enumerate(states)
            put!(cs[state.index],value[i,:])
        end
    end
end


# p=Policy()
## ------------------------------------------------------------
# @spawn listen(p)
# @threads for n=1:4
#     producer(p)
# end
