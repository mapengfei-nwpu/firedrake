from firedrake import *
import pytest
import numpy as np
from mpi4py import MPI

def cell_midpoints(m):
    """Get the coordinates of the midpoints of every cell in mesh `m`.
    The mesh may be distributed, but the midpoints are returned for the
    entire mesh as though it were not distributed."""
    m.init()
    V = VectorFunctionSpace(m, "DG", 0)
    f = Function(V).interpolate(m.coordinates)
    # since mesh may be distributed, the number of cells on the MPI rank
    # may not be the same on all ranks (note we exclude ghost cells
    # hence using num_cells_local = m.cell_set.size). Below local means
    # MPI rank local.
    num_cells_local = m.cell_set.size
    num_cells = MPI.COMM_WORLD.allreduce(num_cells_local, op=MPI.SUM)
    local_midpoints = f.dat.data_ro
    local_midpoints_size = np.array(local_midpoints.size)
    local_midpoints_sizes = np.empty(MPI.COMM_WORLD.size, dtype=int)
    MPI.COMM_WORLD.Allgatherv(local_midpoints_size, local_midpoints_sizes)
    midpoints = np.empty((num_cells, m.cell_dimension()), dtype=float)
    MPI.COMM_WORLD.Allgatherv(local_midpoints, (midpoints, local_midpoints_sizes))
    assert len(np.unique(midpoints, axis=0)) == len(midpoints)
    return midpoints

def _test_pic_swarm_in_plex(m):
    """Generate points in cell midpoints of mesh `m` and check correct
    swarm is created in plex."""
    m.init()
    pointcoords = cell_midpoints(m)
    plex = m.topology._plex
    swarm = mesh._pic_swarm_in_plex(plex, pointcoords)
    # Check comm sizes match
    assert plex.comm.size == swarm.comm.size
    # Get point coords on current MPI rank
    localpointcoords = np.copy(swarm.getField("DMSwarmPIC_coor"))
    swarm.restoreField("DMSwarmPIC_coor")
    if len(pointcoords.shape) > 1:
        localpointcoords = np.reshape(localpointcoords, (-1, pointcoords.shape[1]))
    # check local points are found in list of points
    for p in localpointcoords:
        assert np.any(np.isclose(p, pointcoords))
    # Check methods for checking number of points on current MPI rank
    assert len(localpointcoords) == swarm.getLocalSize()
    # Check there are as many local points as there are local cells
    # (including ghost cells in the halo)
    assert len(localpointcoords) == m.num_cells() == m.cell_set.total_size
    # Check total number of points on all MPI ranks is correct
    # (excluding ghost cells in the halo)
    nghostcellslocal = m.cell_set.total_size - m.cell_set.size
    nghostcellsglobal = MPI.COMM_WORLD.allreduce(nghostcellslocal, op=MPI.SUM)
    nptslocal = len(localpointcoords)
    nptsglobal = MPI.COMM_WORLD.allreduce(nptslocal, op=MPI.SUM)
    assert nptsglobal-nghostcellsglobal == len(pointcoords)
    assert nptsglobal == swarm.getSize()
    # Check each cell has the correct point associated with it
    #TODO

# 1D case not implemented yet
@pytest.mark.parallel(nprocs=2)
def test_pic_swarm_in_plex_1d():
    with pytest.raises(NotImplementedError):
        m = UnitIntervalMesh(1)
        _test_pic_swarm_in_plex(m)

# Need to test cases with 2 cells across 1, 2 and 3 processors
def _test_pic_swarm_in_plex_2d():
    m = UnitSquareMesh(1,1)
    _test_pic_swarm_in_plex(m)

def test_pic_swarm_in_plex_2d(): # nprocs < total number of mesh cells
    _test_pic_swarm_in_plex_2d()

@pytest.mark.parallel(nprocs=2) # nprocs == total number of mesh cells
def test_pic_swarm_in_plex_2d_2procs():
    _test_pic_swarm_in_plex_2d()

@pytest.mark.parallel(nprocs=3) ## nprocs > total number of mesh cells
def test_pic_swarm_in_plex_2d_3procs():
    _test_pic_swarm_in_plex_2d()

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
