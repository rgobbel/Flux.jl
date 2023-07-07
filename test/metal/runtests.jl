Flux.gpu_backend!("Metal")

Metal.allowscalar(false)

# Extend test utils to Metal.

function check_grad(
    g_gpu::MtlArray{Float32}, g_cpu::Array{Float32}; rtol, atol, allow_nothing::Bool,
)
    @test g_cpu ≈ collect(g_gpu) atol=atol rtol=rtol
end

function check_grad(
    g_gpu::MtlArray{Float32}, g_cpu::Zygote.FillArrays.AbstractFill; rtol, atol, allow_nothing::Bool,
)
    @test g_cpu ≈ collect(g_gpu) atol=atol rtol=rtol
end

check_type(x::MtlArray{Float32}) = true

@testset "Basic" begin
    include("basic.jl")
end
