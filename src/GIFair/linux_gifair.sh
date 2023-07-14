echo Run arg: $1

foreach c (0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.5 4.0 4.5 5.0 6.0 7.0 8.0 9.0 10.0 12.0 14.0 16.0 18.0 20.0)
    # foreach s (0 1 2 3 4)
        python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual $c --seed $1 --dataset german --task DP
        python main.py --model LAFTR --fair-coeff $c --fair-coeff-individual 1 --seed $1 --dataset german --task DP

        python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual $c --seed $1 --dataset german --task DP --lambda 10
        python main.py --model LAFTR --fair-coeff $c --fair-coeff-individual 1 --seed $1 --dataset german --task DP --lambda 10

        python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual $c --seed $1 --dataset german --task EO --lambda 10
        python main.py --model LAFTR --fair-coeff $c --fair-coeff-individual 1 --seed $1 --dataset german --task EO --lambda 10

        python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual $c --seed $1 --dataset compas --task DP
        python main.py --model LAFTR --fair-coeff $c --fair-coeff-individual 1 --seed $1 --dataset compas --task DP

        python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual $c --seed $1 --dataset compas --task DP --lambda 10
        python main.py --model LAFTR --fair-coeff $c --fair-coeff-individual 1 --seed $1 --dataset compas --task DP --lambda 10

        python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual $c --seed $1 --dataset compas --task EO --lambda 10
        python main.py --model LAFTR --fair-coeff $c --fair-coeff-individual 1 --seed $1 --dataset compas --task EO --lambda 10

        python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual $c --seed $1 --dataset adult --task DP
        python main.py --model LAFTR --fair-coeff $c --fair-coeff-individual 1 --seed $1 --dataset adult --task DP

        python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual $c --seed $1 --dataset adult --task DP --lambda 10
        python main.py --model LAFTR --fair-coeff $c --fair-coeff-individual 1 --seed $1 --dataset adult --task DP --lambda 10

        python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual $c --seed $1 --dataset adult --task EO --lambda 10
        python main.py --model LAFTR --fair-coeff $c --fair-coeff-individual 1 --seed $1 --dataset adult --task EO --lambda 10
    # end
end

foreach g (0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.5 3.0 3.5 4.0 4.5 5.0)
    # foreach s (0 1 2 3 4)
        python main.py --model LAFTR --fair-coeff 0 --fair-coeff-individual 0 --seed $1 --dataset german --task DP --gamma $g
        python main.py --model LAFTR --fair-coeff 0 --fair-coeff-individual 0 --seed $1 --dataset german --task DP --gamma $g --lambda 10
        python main.py --model LAFTR --fair-coeff 0 --fair-coeff-individual 0 --seed $1 --dataset german --task EO --gamma $g --lambda 10

        python main.py --model LAFTR --fair-coeff 0 --fair-coeff-individual 0 --seed $1 --dataset compas --task DP --gamma $g
        python main.py --model LAFTR --fair-coeff 0 --fair-coeff-individual 0 --seed $1 --dataset compas --task DP --gamma $g --lambda 10
        python main.py --model LAFTR --fair-coeff 0 --fair-coeff-individual 0 --seed $1 --dataset compas --task EO --gamma $g --lambda 10

        python main.py --model LAFTR --fair-coeff 0 --fair-coeff-individual 0 --seed $1 --dataset adult --task DP --gamma $g
        python main.py --model LAFTR --fair-coeff 0 --fair-coeff-individual 0 --seed $1 --dataset adult --task DP --gamma $g --lambda 10
        python main.py --model LAFTR --fair-coeff 0 --fair-coeff-individual 0 --seed $1 --dataset adult --task EO --gamma $g --lambda 10
    # end
end

foreach k (2 4 6 8 10)
    foreach l (0 0.2 0.4 0.6 0.8 1 1.5 2 2.5 3 4 5 6 8 10 15 20)
        # foreach s (0 1 2 3 4)
            python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual 1 --seed $1 --dataset german --task DP --k $k --lambda $l
            python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual 1 --seed $1 --dataset compas --task DP --k $k --lambda $l
            python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual 1 --seed $1 --dataset adult --task DP --k $k --lambda $l
        # end
    end
end

foreach c (0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.5 4.0 4.5 5.0 6.0 7.0 8.0 9.0 10.0 12.0 14.0 16.0 18.0 20.0)
    # foreach s (0 1 2 3 4)
        python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual $c --seed $1 --dataset german --task DP --lambda 10 --aud-steps 0
        python main.py --model LAFTR --fair-coeff $c --fair-coeff-individual 1 --seed $1 --dataset german --task DP --lambda 10 --aud-steps 0

        python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual $c --seed $1 --dataset german --task DP --lambda 10 --aud-individual-steps 0
        python main.py --model LAFTR --fair-coeff $c --fair-coeff-individual 1 --seed $1 --dataset german --task DP --lambda 10 --aud-individual-steps 0

        python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual $c --seed $1 --dataset compas --task DP --lambda 10 --aud-steps 0
        python main.py --model LAFTR --fair-coeff $c --fair-coeff-individual 1 --seed $1 --dataset compas --task DP --lambda 10 --aud-steps 0

        python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual $c --seed $1 --dataset compas --task DP --lambda 10 --aud-individual-steps 0
        python main.py --model LAFTR --fair-coeff $c --fair-coeff-individual 1 --seed $1 --dataset compas --task DP --lambda 10 --aud-individual-steps 0

        python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual $c --seed $1 --dataset adult --task DP --lambda 10 --aud-steps 0
        python main.py --model LAFTR --fair-coeff $c --fair-coeff-individual 1 --seed $1 --dataset adult --task DP --lambda 10 --aud-steps 0

        python main.py --model LAFTR --fair-coeff 1 --fair-coeff-individual $c --seed $1 --dataset adult --task DP --lambda 10 --aud-individual-steps 0
        python main.py --model LAFTR --fair-coeff $c --fair-coeff-individual 1 --seed $1 --dataset adult --task DP --lambda 10 --aud-individual-steps 0
    # end
end