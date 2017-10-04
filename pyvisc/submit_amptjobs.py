#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fri 05 May 2017 06:56:23 PM CEST

from subprocess import call

def submit(collision_system='pbpb2p76', cent='0_5', etaos=0.16,
           jobs_per_gpu=25, ini='ampt', eos_type=1):
    jobs = '''#!/bin/bash                                                                                           
#SBATCH --gres=gpu:4                                                                                  
#SBATCH --constraint=hawaii                                                                           
#SBATCH -D /lustre/nyx/hyihp/lpang/trento_ebe_hydro/PyVisc/pyvisc
#SBATCH --ntasks=4                                                                                    
#SBATCH --cpus-per-task=1                                                                             
#SBATCH --error=log/%a_%j.err                                                                         
#SBATCH --output=log/%a_%j.out                                                                        
#SBATCH --job-name=cl{cent}
#SBATCH --mem-per-cpu=4096                                                                           
#SBATCH --mail-type=ALL                                                                               
#SBATCH --partition=lcsc
#SBATCH --time=48:00:00                                                                               

echo "Start time: $date"

unset DISPLAY
#module load python/2.7                                                                               
#module load amdappsdk/2.9                                                                            
export PATH="/lustre/nyx/hyihp/lpang/anaconda2/bin:$PATH"
export PYTHONPATH="/lustre/nyx/hyihp/lpang/anaconda2/lib/python2.7/:$PYTHONPATH"
export TMPDIR="/lustre/nyx/hyihp/lpang/tmp/"

#modify the cache.py in anaconda/pyopencl and set the cache_dir = TMPDIR

python ebe_{ini}.py '{system}' '{cent}' {etaos} 0 0 50  {eos_type}&
python ebe_{ini}.py '{system}' '{cent}' {etaos} 1 50 100 {eos_type}&
python ebe_{ini}.py '{system}' '{cent}' {etaos} 2 100 150 {eos_type}&
python ebe_{ini}.py '{system}' '{cent}' {etaos} 3 150 200 {eos_type}&

wait
echo "End time: $date"
'''.format(system=collision_system, cent=cent, ini=ini, eos_type=eos_type, etaos=etaos)

    job_name = "gsi_cent%s_%s.sh"%(cent, ini)
    with open(job_name, 'w') as fout:
        fout.write(jobs)
    call(['sbatch', job_name])
    call(['mv', job_name, 'jobs/'])

# comparing the grid size dependence of the hypersf cube
# etaos_ymin = 0.08 for both ampt runs, not 0.2

if __name__=='__main__':
    submit(cent='0_5')
    submit(cent='5_10')
    submit(cent='10_20')
    submit(cent='20_30')
    submit(cent='30_40')
    submit(cent='40_50')
    submit(cent='50_60')
