#!/bin/bash                                                                                           
#SBATCH --gres=gpu:4                                                                                  
#SBATCH --constraint=hawaii                                                                           
#SBATCH -D /lustre/nyx/hyihp/lpang/PyVisc/pbpb5p02/
#SBATCH --ntasks=4                                                                                    
#SBATCH --cpus-per-task=1                                                                             
#SBATCH --error=log/%a_%j.err                                                                         
#SBATCH --output=log/%a_%j.out                                                                        
#SBATCH --job-name=clvisc
#SBATCH --mem-per-cpu=4096                                                                           
#SBATCH --mail-type=ALL                                                                               
#SBATCH --partition=lcsc
#SBATCH --time=2:00:00                                                                               

echo "Start time: $date"

unset DISPLAY
#module load python/2.7                                                                               
#module load amdappsdk/2.9                                                                            

export PATH="/lustre/nyx/hyihp/lpang/anaconda/bin:$PATH"
export PYTHONPATH="/lustre/nyx/hyihp/lpang/anaconda/lib/python/:$PYTHONPATH"
export TMPDIR="/lustre/nyx/hyihp/lpang/tmp/"

#modify the cache.py in anaconda/pyopencl and set the cache_dir = TMPDIR

python pbpb.py 86 100 0 5 0 &
python pbpb.py 86 101 5 10 1 &
python pbpb.py 86 102 10 20 2 &
python pbpb.py 86 103 20 30 3 &

wait 

python pbpb.py 86 104 0 10 0 &
python pbpb.py 86 105 0 80 1 &
python pbpb.py 86 106 10 30 2 &
python pbpb.py 86 107 30 50 3 &
wait 


echo "End time: $date"
