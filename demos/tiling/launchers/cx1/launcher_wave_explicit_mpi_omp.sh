#!/bin/bash

#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=20:mem=16gb:icib=true
#PBS -q pqcdt

# Note: mpiexec.1pps runs hybrid mpi+openmp jobs
# Behind the scenes, it sets:
# I_MPI_PIN=yes
# I_MPI_PIN_MODE=lib
# I_MPI_PIN_DOMAIN=socket
# I_MPI_PIN_ORDER=compact
# KMP_AFFINITY=granularity=fine,compact,1,0

FIREDRAKE=$HOME/Projects/Firedrake/firedrake
TILING=$FIREDRAKE/demos/tiling
EXECUTABLE=$TILING/wave_explicit_fusion.py
MESHES=$TILING/meshes/wave_explicit


echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $PBS_NODEFILE
cat /proc/cpuinfo | grep "model name" | uniq
echo ------------------------------------------------------
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: originating queue is $PBS_O_QUEUE
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: execution mode is $PBS_ENVIRONMENT
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH
echo ------------------------------------------------------
echo PBS: PYTHONPATH = $PYTHONPATH
echo ------------------------------------------------------
echo PBS: SLOPE_BACKEND = $SLOPE_BACKEND
echo ------------------------------------------------------


# Clean the remote cache, then dry runs on tiny mesh to generate kernels
$FIREDRAKE/scripts/firedrake-clean
for nu in 0 1 2
do
    for p in "chunk" "metis"
    do
        for m in 10
        do
            for ts in 4
            do
                # OMP backends:
                export SLOPE_BACKEND=OMP
                # ... pure openmp
                python $EXECUTABLE --mesh-file $m --tile-size $ts --num-unroll $nu
                # ... hybrid mpi-openmp
                mpiexec.1pps python $EXECUTABLE --mesh-file $m --tile-size $ts --part-mode $p --num-unroll $nu

                # MPI backend:
                export SLOPE_BACKEND=SEQUENTIAL
                mpiexec python $EXECUTABLE --mesh-file $m --tile-size $ts --part-mode $p --num-unroll $nu
            done
        done
    done
done


# Run the tests
for nu in 0 1 2
do
    for p in "chunk" "metis"
    do
        for m in $MESHES"/wave_tank_0.125.msh"
        do
            for ts in 500 2000 3000 5000
            do
                # OMP backends:
                export SLOPE_BACKEND=OMP
                # ... pure openmp
                python $EXECUTABLE --mesh-file $m --tile-size $ts --num-unroll $nu
                # ... hybrid mpi-openmp
                mpiexec.1pps python $EXECUTABLE --mesh-file $m --tile-size $ts --part-mode $p --num-unroll $nu

                # MPI backend:
                export SLOPE_BACKEND=SEQUENTIAL
                mpiexec python $EXECUTABLE --mesh-file $m --tile-size $ts --part-mode $p --num-unroll $nu
            done
        done
    done
done
