from firedrake import *
import pytest
import numpy as np
from mpi4py import MPI

def cell_midpoints(m):
    V = VectorFunctionSpace(m, "DG", 0)
    f = Function(V).interpolate(m.coordinates)
    # since mesh may be distributed, the number of cells may not be the same on
    # all ranks
    num_cells = MPI.COMM_WORLD.allreduce(m.cell_set.size, op=MPI.SUM)
    midpoints = np.empty((num_cells, m.cell_dimension()), dtype=float)
    local_midpoints = f.dat.data_ro
    MPI.COMM_WORLD.Allgatherv(local_midpoints, midpoints)
    return midpoints

def _test_pic_swarm_in_plex(m):
    """Generate points in cell midpoints of mesh `m` and check correct
    swarm is created in plex."""
    pointcoords = cell_midpoints(m)
    assert len(np.unique(pointcoords, axis=0)) == len(pointcoords)
    plex = m.topology._plex
    swarm = mesh._pic_swarm_in_plex(plex, pointcoords)
    # Get point coords on current MPI rank
    localpointcoords = np.copy(swarm.getField("DMSwarmPIC_coor"))
    swarm.restoreField("DMSwarmPIC_coor")
    # check local points are found in list of points
    localpointcoords = np.reshape(localpointcoords, (-1, pointcoords.shape[1]))
    for p in localpointcoords:
        assert np.any(np.isclose(p, pointcoords))
    # Check methods for checking number of points on current MPI rank
    assert len(localpointcoords) == swarm.getLocalSize()
    # Check there are as many local points as there are local cells
    assert len(localpointcoords) == m.cell_set.size
    # Check total number of points on all MPI ranks is correct
    nptslocal = len(localpointcoords)
    nptsglobal = MPI.COMM_WORLD.allreduce(nptslocal, op=MPI.SUM)
    assert nptsglobal == len(pointcoords)
    assert nptsglobal == swarm.getSize()
    # Check each cell has the correct point associated with it
    #TODO

# 1D case not implemented yet
@pytest.mark.parallel(nprocs=2)
def test_pic_swarm_in_plex_1d():
    with pytest.raises(NotImplementedError):
        m = UnitIntervalMesh(1)
        _test_pic_swarm_in_plex(m)

@pytest.mark.parallel(nprocs=2)
def test_pic_swarm_in_plex_2d():
    m = UnitSquareMesh(1,1)
    _test_pic_swarm_in_plex(m)

@pytest.mark.parallel(nprocs=2)
def test_pic_swarm_in_plex_3d():
    m = UnitCubeMesh(1,1,1)
    _test_pic_swarm_in_plex(m)

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
