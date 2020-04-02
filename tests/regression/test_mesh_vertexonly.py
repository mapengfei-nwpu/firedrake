from firedrake import *
import pytest
import numpy as np
from mpi4py import MPI

def cell_midpoints(m):
    V = VectorFunctionSpace(m, "DG", 0)
    f = Function(V).interpolate(m.coordinates)
    midpoints = f.dat.data_ro
    # since mesh may be distributed, midpoints may not be the same on
    # all ranks. Therefore need to gather...
    all_midpoints = np.empty((m.num_cells(), m.cell_dimension()), dtype=float)
    # MPI.COMM_WORLD.Allgather(midpoints, all_midpoints)
    all_midpoints = MPI.COMM_WORLD.allgather(f.dat.data)
    # below will fail if there are empty entries
    return np.asarray(all_midpoints)

def _test_pic_swarm_in_plex(m, points):
    plex = m.topology._plex
    swarm = mesh._pic_swarm_in_plex(plex, points)
    # Check how many points are on current MPI rank
    nlocal = swarm.getLocalSize()
    # Check total number of points on all MPI ranks is correct
    nglobal = MPI.COMM_WORLD.allreduce(nlocal, op=MPI.SUM)
    assert nglobal == len(points)
    assert nglobal == swarm.getSize()
    # Check each cell has the correct point associated with it
    #TODO

# 1D case not implemented yet
@pytest.mark.parallel(nprocs=2)
def test_pic_swarm_in_plex_1d():
    with pytest.raises(NotImplementedError):
        m = UnitIntervalMesh(1)
        _test_pic_swarm_in_plex(m, cell_midpoints(m))

@pytest.mark.parallel(nprocs=2)
def test_pic_swarm_in_plex_2d():
    m = UnitSquareMesh(1,1)
    _test_pic_swarm_in_plex(m, cell_midpoints(m))

@pytest.mark.parallel(nprocs=2)
def test_pic_swarm_in_plex_3d():
    m = UnitCubeMesh(1,1,1)
    _test_pic_swarm_in_plex(m, cell_midpoints(m))

def verify_vertexonly_mesh(m, vm):
    # test that the mesh properties are as expected
    vm.init()
    #TODO

def test_generate():

    # 1D case not implemented yet
    with pytest.raises(NotImplementedError):
        m = UnitIntervalMesh(1)
        vertexcoords = cell_midpoints(m)
        vm = VertexOnlyMesh(m, vertexcoords)
        verify_vertexonly_mesh(m, vm)

    m = UnitSquareMesh(1,1)
    vertexcoords = cell_midpoints(m)
    vm = VertexOnlyMesh(m, vertexcoords)
    verify_vertexonly_mesh(m, vm)

    m = UnitCubeMesh(1,1,1)
    vertexcoords = cell_midpoints(m)
    vm = VertexOnlyMesh(m, vertexcoords)
    verify_vertexonly_mesh(m, vm)

# remove this before final merge
if __name__ == "__main__":
    import pytest, sys
    pytest.main([sys.argv[0]])
