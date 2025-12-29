using Test, PreallocationTools, ForwardDiff

# Test DiffCache with complex arrays (issue #47)
@testset "DiffCache with Complex Arrays" begin
    # 1D complex array test
    @testset "1D ComplexF64 array" begin
        z = zeros(ComplexF64, 20)
        zd = DiffCache(z)

        function sum_cis_1d(zd)
            return function (θ)
                z = get_tmp(zd, θ)
                for i in eachindex(z)
                    z[i] = cis(i * θ)
                end
                abs(sum(z))
            end
        end

        f = sum_cis_1d(zd)
        result = ForwardDiff.derivative(f, 1.1)

        # Verify numerically
        h = 1e-7
        numerical = (f(1.1 + h) - f(1.1 - h)) / (2h)
        @test isapprox(result, numerical, rtol = 1e-5)

        # Non-AD case should return complex array
        z_normal = get_tmp(zd, 1.0)
        @test eltype(z_normal) == ComplexF64
        @test size(z_normal) == size(z)
    end

    # 2D complex array test
    @testset "2D ComplexF64 array" begin
        z = zeros(ComplexF64, 4, 5)
        zd = DiffCache(z)

        function sum_cis_2d(zd)
            return function (θ)
                z = get_tmp(zd, θ)
                for i in eachindex(z)
                    z[i] = cis(i * θ)
                end
                abs(sum(z))
            end
        end

        f = sum_cis_2d(zd)
        result = ForwardDiff.derivative(f, 1.1)

        h = 1e-7
        numerical = (f(1.1 + h) - f(1.1 - h)) / (2h)
        @test isapprox(result, numerical, rtol = 1e-5)

        # Check dimensions preserved
        z_ad = get_tmp(zd, ForwardDiff.Dual(1.0, 1.0))
        @test size(z_ad) == size(z)
    end

    # ComplexF32 test
    @testset "ComplexF32 array" begin
        z = zeros(ComplexF32, 10)
        zd = DiffCache(z)

        function sum_cis_f32(zd)
            return function (θ)
                z = get_tmp(zd, θ)
                for i in eachindex(z)
                    z[i] = cis(Float32(i) * θ)
                end
                abs(sum(z))
            end
        end

        f = sum_cis_f32(zd)
        result = ForwardDiff.derivative(f, 1.1f0)

        h = 1.0f-5
        numerical = (f(1.1f0 + h) - f(1.1f0 - h)) / (2h)
        @test isapprox(result, numerical, rtol = 1e-3)
    end
end

@testset "FixedSizeDiffCache with Complex Arrays" begin
    # 1D complex array test
    @testset "1D ComplexF64 array" begin
        z = zeros(ComplexF64, 20)
        chunk_size = 5
        zd = FixedSizeDiffCache(z, chunk_size)

        function sum_cis_fixed_1d(zd)
            return function (θ)
                z = get_tmp(zd, θ)
                for i in eachindex(z)
                    z[i] = cis(i * θ)
                end
                abs(sum(z))
            end
        end

        f = sum_cis_fixed_1d(zd)
        result = ForwardDiff.derivative(f, 1.1)

        h = 1e-7
        numerical = (f(1.1 + h) - f(1.1 - h)) / (2h)
        @test isapprox(result, numerical, rtol = 1e-5)

        # Non-AD case
        z_normal = get_tmp(zd, 1.0)
        @test eltype(z_normal) == ComplexF64
        @test size(z_normal) == size(z)
    end

    # 2D complex array test
    @testset "2D ComplexF64 array" begin
        z = zeros(ComplexF64, 4, 5)
        chunk_size = 3
        zd = FixedSizeDiffCache(z, chunk_size)

        function sum_cis_fixed_2d(zd)
            return function (θ)
                z = get_tmp(zd, θ)
                for i in eachindex(z)
                    z[i] = cis(i * θ)
                end
                abs(sum(z))
            end
        end

        f = sum_cis_fixed_2d(zd)
        result = ForwardDiff.derivative(f, 1.1)

        h = 1e-7
        numerical = (f(1.1 + h) - f(1.1 - h)) / (2h)
        @test isapprox(result, numerical, rtol = 1e-5)

        # Check dimensions preserved
        z_ad = get_tmp(zd, ForwardDiff.Dual(1.0, 1.0))
        @test size(z_ad) == size(z)
    end
end

@testset "Complex DiffCache with gradient computation" begin
    # Test with ForwardDiff.gradient
    z = zeros(ComplexF64, 5)
    zd = DiffCache(z)

    function complex_quadratic(zd)
        return function (x)
            z = get_tmp(zd, x)
            for i in eachindex(z)
                z[i] = x[i] * cis(x[i])
            end
            real(sum(z .* conj.(z)))
        end
    end

    f = complex_quadratic(zd)
    x0 = [1.0, 2.0, 3.0, 4.0, 5.0]
    grad = ForwardDiff.gradient(f, x0)

    # Verify with numerical gradient
    h = 1e-7
    numerical_grad = similar(grad)
    for i in eachindex(x0)
        x_plus = copy(x0)
        x_plus[i] += h
        x_minus = copy(x0)
        x_minus[i] -= h
        numerical_grad[i] = (f(x_plus) - f(x_minus)) / (2h)
    end

    @test isapprox(grad, numerical_grad, rtol = 1e-5)
end
