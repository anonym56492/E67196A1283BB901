echo Run arg: %1

for %%c in (0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.5 4.0 4.5 5.0 6.0 7.0 8.0 9.0 10.0 12.0 14.0 16.0 18.0 20.0)
    rem for %%s in (0 1 2 3 4) do (
        python main.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset german --task DP

        python main.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset german --task DP --lambda 10

        python main.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset german --task EO --lambda 10

        python main.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset compas --task DP

        python main.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset compas --task DP --lambda 10

        python main.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset compas --task EO --lambda 10

        python main.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset adult --task DP

        python main.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset adult --task DP --lambda 10

        python main.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset adult --task EO --lambda 10
        
        python evaluate.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset german --task DP

        python evaluate.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset german --task DP --lambda 10

        python evaluate.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset german --task EO --lambda 10

        python evaluate.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset compas --task DP

        python evaluate.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset compas --task DP --lambda 10

        python evaluate.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset compas --task EO --lambda 10

        python evaluate.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset adult --task DP

        python evaluate.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset adult --task DP --lambda 10

        python evaluate.py --model LAFTR --fair-coeff-individual %%c --seed %1 --dataset adult --task EO --lambda 10
    rem )
end