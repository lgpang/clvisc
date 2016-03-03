#!/bin/bash                                                                                           
#SBATCH --gres=gpu:2                                                                                  
#SBATCH --constraint=hawaii                                                                           
#SBATCH -D /lustre/nyx/hyihp/lpang/auau39_results/
#SBATCH --ntasks=1                                                                                    
#SBATCH --cpus-per-task=1                                                                             
#SBATCH --error=log/%a_%j.err                                                                         
#SBATCH --output=log/%a_%j.out                                                                        
#SBATCH --job-name=clvisc
#SBATCH --mem-per-cpu=4096                                                                           
#SBATCH --mail-type=ALL                                                                               
#SBATCH --partition=lcsc                                                                              
#SBATCH --time=24:00:00                                                                               

echo "Start time: $date"

unset DISPLAY
#module load python/2.7                                                                               
#module load amdappsdk/2.9                                                                            

python ebe_hdf5_auau39.py

echo "End time: $date"
