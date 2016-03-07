#!/bin/bash                                                                                           
#SBATCH --gres=gpu:4                                                                                  
#SBATCH --constraint=hawaii                                                                           
#SBATCH -D /lustre/nyx/hyihp/lpang/PyVisc/bin/
#SBATCH --ntasks=4                                                                                    
#SBATCH --cpus-per-task=1                                                                             
#SBATCH --error=log/%a_%j.err                                                                         
#SBATCH --output=log/%a_%j.out                                                                        
#SBATCH --job-name=clvisc
#SBATCH --mem-per-cpu=4096                                                                           
#SBATCH --mail-type=ALL                                                                               
#SBATCH --partition=lcsc                                                                              
#SBATCH --time=48:00:00                                                                               

echo "Start time: $date"

unset DISPLAY
#module load python/2.7                                                                               
#module load amdappsdk/2.9                                                                            

export PATH="/lustre/nyx/hyihp/lpang/anaconda/bin:$PATH"
export PYTHONPATH="/lustre/nyx/hyihp/lpang/anaconda/lib/python/:$PYTHONPATH"
export TMPDIR="/lustre/nyx/hyihp/lpang/tmp/"

#modify the cache.py in anaconda/pyopencl and set the cache_dir = TMPDIR

python ebe.py auau200 20_50 0.08 0 &
python ebe.py auau62p4 20_50 0.08 1 &
python ebe.py auau39 20_50 0.08 2 &
python ebe.py auau19p6 20_50 0.08 3 &
wait

echo "End time: $date"
