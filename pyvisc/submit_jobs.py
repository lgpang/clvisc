#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fri 05 May 2017 06:56:23 PM CEST

from subprocess import call

def submit(cent='0_5', jobs_per_gpu=25):
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
#SBATCH --time=24:00:00                                                                               

echo "Start time: $date"

unset DISPLAY
#module load python/2.7                                                                               
#module load amdappsdk/2.9                                                                            
export PATH="/lustre/nyx/hyihp/lpang/anaconda2/bin:$PATH"
export PYTHONPATH="/lustre/nyx/hyihp/lpang/anaconda2/lib/python2.7/:$PYTHONPATH"
export TMPDIR="/lustre/nyx/hyihp/lpang/tmp/"

#modify the cache.py in anaconda/pyopencl and set the cache_dir = TMPDIR

python ebe.py '{cent}' 0 {jobs_per_gpu} &
python ebe.py '{cent}' 1 {jobs_per_gpu} &
python ebe.py '{cent}' 2 {jobs_per_gpu} &
python ebe.py '{cent}' 3 {jobs_per_gpu} &

wait
echo "End time: $date"
'''.format(cent=cent, jobs_per_gpu=jobs_per_gpu)

    job_name = "gsi_cent%s.sh"%cent
    with open(job_name, 'w') as fout:
        fout.write(jobs)
    call(['sbatch', job_name])
    call(['mv', job_name, 'jobs/'])


if __name__=='__main__':
    submit(cent='0_5')
    #submit(cent='5_10')
    #submit(cent='10_20')
    #submit(cent='20_30')
