import pytest
import numpy as np

from firedrake import *

@pytest.mark.parallel(nprocs=2)
def test_pic_swarm_in_plex_1d_2procs():
    from mpi4py import MPI
    # Mesh with two cells
    m = UnitIntervalMesh(2)
    # 4 points such that there are two per cell
    points = [(.1,), (.2,), (.8,), (.9,)]
    plex = m.topology._plex
    swarm = mesh._pic_swarm_in_plex(plex, points)
    # Check how many points are on current MPI rank
    nlocal = swarm.getLocalSize()
    # Check total number of points on all MPI ranks is correct
    nglobal = MPI.COMM_WORLD.allreduce(nlocal, op=MPI.SUM)
    assert nglobal == len(points)
    assert nglobal == swarm.getSize()
    # Now check that we have two points in each cell as expected
    assert nlocal == 2

@pytest.mark.parallel(nprocs=2)
def test_pic_swarm_in_plex_2d_2procs():
    from mpi4py import MPI
    # Mesh with two cells
    m = UnitSquareMesh(1,1)
    # 4 points such that there are two per cell
    points = [(.1, .1), (.2, .2), (.8, .8), (.9, .9)]
    plex = m.topology._plex
    swarm = mesh._pic_swarm_in_plex(plex, points)
    # Check how many points are on current MPI rank
    nlocal = swarm.getLocalSize()
    # Check total number of points on all MPI ranks is correct
    nglobal = MPI.COMM_WORLD.allreduce(nlocal, op=MPI.SUM)
    assert nglobal == len(points)
    assert nglobal == swarm.getSize()
    # Now check that we have two points in each cell as expected
    assert nlocal == 2

def verify_vertexonly_mesh(m, vm):
    # test that the mesh properties are as expected
    vm.init()
    #TODO

def test_generate():
    m = UnitSquareMesh(5,5)
    vertexcoords = [(.1, .1), (.2, .3), (.7, .8)]
    vm = VertexOnlyMesh(m, vertexcoords)
    verify_vertexonly_mesh(m, vm)

# remove this before final merge
if __name__ == "__main__":
    import pytest, sys
    pytest.main([sys.argv[0]])
