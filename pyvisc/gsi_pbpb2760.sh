#!/bin/bash                                                                                           
#SBATCH --gres=gpu:4                                                                                  
#SBATCH --constraint=hawaii                                                                           
#SBATCH -D /lustre/nyx/hyihp/lpang/trento_ebe_hydro/PyVisc/pyvisc
#SBATCH --ntasks=4                                                                                    
#SBATCH --cpus-per-task=1                                                                             
#SBATCH --error=log/%a_%j.err                                                                         
#SBATCH --output=log/%a_%j.out                                                                        
#SBATCH --job-name=cl1040
#SBATCH --mem-per-cpu=4096                                                                           
#SBATCH --mail-type=ALL                                                                               
#SBATCH --partition=lcsc
#SBATCH --time=24:00:00                                                                               

echo "Start time: $date"

unset DISPLAY
#module load python/2.7                                                                               
#module load amdappsdk/2.9                                                                            
export PATH="/lustre/nyx/hyihp/lpang/anaconda2/bin:$PATH"
export PYTHONPATH="/lustre/nyx/hyihp/lpang/anaconda2/lib/python2.7/:$PYTHONPATH"
export TMPDIR="/lustre/nyx/hyihp/lpang/tmp/"

#modify the cache.py in anaconda/pyopencl and set the cache_dir = TMPDIR

python ebe.py '0_5' 0 25 &
python ebe.py '0_5' 1 25 &
python ebe.py '0_5' 2 25 &
python ebe.py '0_5' 3 25 &

wait

echo "End time: $date"
