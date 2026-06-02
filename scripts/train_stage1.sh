echo "Let's run some Experiments"

for i in 1 2 3 4 5; 
do
    echo "Running Experiment: $i"
    sbatch ./experiments/exp$i.sh
done