@testset "Endgame AD: $AD" for AD = 3:4
    @testset "Cyclic 7" begin
        f = cyclic(7)
        H, starts = total_degree_homotopy(f)
        S = collect(starts)
        tracker = PathTracker(Tracker(H, automatic_differentiation = AD))
        res = track.(tracker, S)

        @test count(is_success, res) == 924
        @test 4000 ≤ count(is_at_infinity, res) ≤ 4116
    end

    @testset "Hyperbolic - 6,6" begin
        # 2 solutions with multiplicity 6, projective
        @var x z
        y = 1
        # This has two roots of multiplicity 6 at the hyperplane z=0
        # each root has winding number 3
        F = [
            0.75 * x^4 + 1.5 * x^2 * y^2 - 2.5 * x^2 * z^2 + 0.75 * y^4 - 2.5 * y^2 * z^2 + 0.75 * z^4
            10 * x^2 * z + 10 * y^2 * z - 6 * z^3
        ]
        H, starts = total_degree_homotopy(F, [x, z])
        S = collect(starts)
        tracker = PathTracker(Tracker(H, automatic_differentiation = AD))
        res = track.(tracker, S)
        @test count(r -> r.winding_number == 3, res) == 12
    end

    @testset "Wilkinson 19" begin
        @var x
        d = 19
        f = expand(prod(x - i for i = 1:d))
        H, starts = total_degree_homotopy([f], [x], gamma = 0.4 + 1.3im)
        S = collect(starts)
        tracker = PathTracker(Tracker(H, automatic_differentiation = AD))
        res = track.(tracker, S)
        @test all(is_success, res)
        @test round.(Int, real.(sort(first.(solution.(res)); by = abs))) == 1:19
        @test maximum(abs.(imag.(first.(solution.(res))))) < 1e-8
        @test count(r -> isnothing(r.winding_number), res) == 19
    end

    @testset "(x-10)^$d" for d in [2, 8, 12, 16, 18 ]
        @var x
        f = [(x - 10)^d]
        H, starts = total_degree_homotopy(f, [x])
        S = collect(starts)
        tracker = PathTracker(Tracker(H, automatic_differentiation = AD))
        res = track.(tracker, S)
        @test count(r -> r.winding_number == d, res) == d
    end

    @testset "Beyond Polyhedral Homotopy Example" begin
        @var x y
        f = [2.3 * x^2 + 1.2 * y^2 + 3x - 2y + 3, 2.3 * x^2 + 1.2 * y^2 + 5x + 2y - 5]
        H, starts = total_degree_homotopy(f, [x, y]; gamma = 1.3im + 0.4)
        S = collect(starts)
        tracker = PathTracker(Tracker(H, automatic_differentiation = AD))
        res = track.(tracker, S)
        @test count(is_success, res) == 2
        @test count(is_at_infinity, res) == 2
    end

    @testset "Winding Number Family d=$d" for d = 2:2:6
        @var x y
        a = [0.257, -0.139, -1.73, -0.199, 1.79, -1.32]
        f1 = (a[1] * x^d + a[2] * y) * (a[3] * x + a[4] * y) + 1
        f2 = (a[1] * x^d + a[2] * y) * (a[5] * x + a[6] * y) + 1
        H, starts = total_degree_homotopy([f1, f2], [x, y]; gamma = 1.3im + 0.4)
        S = collect(starts)
        tracker = PathTracker(Tracker(H, automatic_differentiation = AD))
        res = track.(tracker, S)
        @test count(is_success, res) == d + 1
        @test count(is_at_infinity, res) == (d + 1)^2 - d - 1
    end

    @testset "Bacillus Subtilis" begin
        H, starts = total_degree_homotopy(bacillus())
        tracker = PathTracker(Tracker(H, automatic_differentiation = AD))
        res = track.(tracker, starts)
        @test count(is_success, res) == 44
    end

    @testset "Mohab" begin
        # Communicated by Mohab Safey El Din
        @var x y z

        F =
            ModelKit.horner.([
                -9091098778555951517 * x^3 * y^4 * z^2 +
                5958442613080401626 * y^2 * z^7 +
                17596733865548170996 * x^2 * z^6 - 17979170986378486474 * x * y * z^6 -
                2382961149475678300 * x^4 * y^3 - 15412758154771986214 * x * y^3 * z^3 +
                133,
                -10798198881812549632 * x^6 * y^3 * z - 11318272225454111450 * x * y^9 -
                14291416869306766841 * y^9 * z - 5851790090514210599 * y^2 * z^8 +
                15067068695242799727 * x^2 * y^3 * z^4 +
                7716112995720175148 * x^3 * y * z^3 +
                171,
                13005416239846485183 * x^7 * y^3 + 4144861898662531651 * x^5 * z^4 -
                8026818640767362673 * x^6 - 6882178109031199747 * x^2 * y^4 +
                7240929562177127812 * x^2 * y^3 * z +
                5384944853425480296 * x * y * z^4 +
                88,
            ])

        H, starts = total_degree_homotopy(System(F, [x, z, y]))
        tracker = PathTracker(Tracker(H, automatic_differentiation = AD))
        res = track.(tracker, starts)
        @test count(is_success, res) == 693
    end
end
