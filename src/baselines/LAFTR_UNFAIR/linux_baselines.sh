echo Run arg: $1

foreach c (0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.5 4.0 4.5 5.0 6.0 7.0 8.0 9.0 10.0 12.0 14.0 16.0 18.0 20.0)
    # foreach s (0 1 2 3 4)
        python main.py --model LAFTR --fair-coeff $c --seed $1 --dataset german --task DP --aud-steps 1
        python main.py --model UNFAIR --seed $1 --dataset german --aud-steps 1

        python main.py --model LAFTR --fair-coeff $c --seed $1 --dataset compas --task DP --aud-steps 1
        python main.py --model UNFAIR --seed $1 --dataset compas --aud-steps 1

        python main.py --model LAFTR --fair-coeff $c --seed $1 --dataset adult --task DP --aud-steps 1
        python main.py --model UNFAIR --seed $1 --dataset adult --aud-steps 1
        
        python evaluate.py --model LAFTR --fair-coeff $c --seed $1 --dataset german --task DP --aud-steps 1
        python evaluate.py --model UNFAIR --seed $1 --dataset german --aud-steps 1

        python evaluate.py --model LAFTR --fair-coeff $c --seed $1 --dataset compas --task DP --aud-steps 1
        python evaluate.py --model UNFAIR --seed $1 --dataset compas --aud-steps 1

        python evaluate.py --model LAFTR --fair-coeff $c --seed $1 --dataset adult --task DP --aud-steps 1
        python evaluate.py --model UNFAIR --seed $1 --dataset adult --aud-steps 1
		
        python main.py --model LAFTR --fair-coeff $c --seed $1 --dataset german --task EO --aud-steps 1
        python main.py --model LAFTR --fair-coeff $c --seed $1 --dataset compas --task EO --aud-steps 1
        python main.py --model LAFTR --fair-coeff $c --seed $1 --dataset adult --task EO --aud-steps 1

        python evaluate.py --model LAFTR --fair-coeff $c --seed $1 --dataset german --task EO --aud-steps 1
        python evaluate.py --model LAFTR --fair-coeff $c --seed $1 --dataset compas --task EO --aud-steps 1
        python evaluate.py --model LAFTR --fair-coeff $c --seed $1 --dataset adult --task EO --aud-steps 1
    # end
end