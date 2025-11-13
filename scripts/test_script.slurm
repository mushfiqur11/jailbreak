#!/bin/sh

#SBATCH --job-name=run_session

#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --output=/scratch/%u/jailbreaking_repos/outputs/test_unit.out # Output file
#SBATCH --error=/scratch/%u/jailbreaking_repos/outputs/test_unit.err # Error file
#SBATCH --mail-type=FAIL    # ALL, NONE, BEGIN, END, FAIL, REQUEUE,..
#SBATCH --mail-user=mrahma45@gmu.edu # Put your GMU email address here

#SBATCH --mem=100GB # Total memory needed per task (units: K,M,G,T)
#SBATCH --time=0-10:00 # Total time needed for job: Days-Hours:Minutes

#SBATCH --cpus-per-task 1 # Request extra CPUs for threads
#SBATCH --ntasks 1 # Number of processes you plan to launch
#SBATCH --gres=gpu:A100.40gb:1

## Load the relevant modules needed for the job

set echo 
umask 0022 

## Run your program or script
## Keep the job running for 10 hours
module load gnu10

cd /scratch/mrahma45/jailbreaking_repos/
source venv_jailbreak/bin/activate

cd jailbreak-pkg/runs

# python llm_module/basic_usage.py
python unit_tests/test_parsers.py