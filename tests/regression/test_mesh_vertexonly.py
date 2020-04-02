import pytest
import numpy as np

from firedrake import *


# 1D case not implemented yet
# @pytest.mark.parallel(nprocs=2)
# def test_pic_swarm_in_plex_1d_2procs():
#     from mpi4py import MPI
#     # Mesh with two cells
#     m = UnitIntervalMesh(2)
#     # 4 points such that there are two per cell
#     points = [(.1,), (.2,), (.8,), (.9,)]
#     plex = m.topology._plex
#     swarm = mesh._pic_swarm_in_plex(plex, points)
#     # Check how many points are on current MPI rank
#     nlocal = swarm.getLocalSize()
#     # Check total number of points on all MPI ranks is correct
#     nglobal = MPI.COMM_WORLD.allreduce(nlocal, op=MPI.SUM)
#     assert nglobal == len(points)
#     assert nglobal == swarm.getSize()
#     # Now check that we have two points in each cell as expected
#     assert nlocal == 2

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

# TODO
# @pytest.mark.parallel(nprocs=6)
# def test_pic_swarm_in_plex_3d_6procs():
#     from mpi4py import MPI
#     # Mesh with six cells
#     m = UnitCubeMesh(1,1,1)
#     # 6 points such that there is one per cell
#     # points = [(.5, .5, 0.), (.5, .5, 1.), (.5, 0., .5), (.5, 1., .5), (0., .5, .5), (1., .5, .5)]
#     points = [ (0.66666667, 0.33333333, 0.        ),
#                (1.        , 0.66666667, 0.33333333),
#                (0.66666667, 1.        , 0.33333333),
#                (0.66666667, 0.66666667, 0.66666667),
#                (0.33333333, 0.66666667, 0.66666667),
#                (0.33333333, 0.33333333, 1.        ) ]
#     plex = m.topology._plex
#     swarm = mesh._pic_swarm_in_plex(plex, points)
#     # Check how many points are on current MPI rank
#     nlocal = swarm.getLocalSize()
#     # Now check that we have one point in each cell as expected
#     assert nlocal == 1
#     # Check total number of points on all MPI ranks is correct
#     nglobal = MPI.COMM_WORLD.allreduce(nlocal, op=MPI.SUM)
#     assert nglobal == len(points)
#     assert nglobal == swarm.getSize()

def verify_vertexonly_mesh(m, vm):
    # test that the mesh properties are as expected
    vm.init()
    #TODO

def test_generate():

    with pytest.raises(NotImplementedError):
        m = UnitIntervalMesh(5)
        vertexcoords = [(.1,), (.2,), (.7,)]
        vm = VertexOnlyMesh(m, vertexcoords)
        verify_vertexonly_mesh(m, vm)

    m = UnitSquareMesh(5,5)
    vertexcoords = [(.1, .1), (.2, .3), (.7, .8)]
    vm = VertexOnlyMesh(m, vertexcoords)
    verify_vertexonly_mesh(m, vm)

    # TODO
    # m = UnitCubeMesh(2,2,2)
    # vertexcoords = [(.1, .1, .1), (.2, .3, .2), (.7, .8, .7)]
    # vm = VertexOnlyMesh(m, vertexcoords)
    # verify_vertexonly_mesh(m, vm)

# remove this before final merge
if __name__ == "__main__":
    import pytest, sys
    pytest.main([sys.argv[0]])
