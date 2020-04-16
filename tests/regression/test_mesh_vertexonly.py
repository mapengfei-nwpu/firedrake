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
    return swarm

# 1D case not implemented yet
@pytest.mark.xfail
def test_pic_swarm_in_plex_1d():
    swarm = _test_pic_swarm_in_plex(UnitIntervalMesh(1))

# Need to test cases with 2 cells across 1, 2 and 3 processors
def _test_pic_swarm_in_plex_2d():
    swarm = _test_pic_swarm_in_plex(UnitSquareMesh(1,1))

def test_pic_swarm_in_plex_2d(): # nprocs < total number of mesh cells
    _test_pic_swarm_in_plex_2d()

@pytest.mark.xfail
@pytest.mark.parallel(nprocs=2) # nprocs == total number of mesh cells
def test_pic_swarm_in_plex_2d_2procs():
    _test_pic_swarm_in_plex_2d()

@pytest.mark.xfail
@pytest.mark.parallel(nprocs=3) ## nprocs > total number of mesh cells
def test_pic_swarm_in_plex_2d_3procs():
    _test_pic_swarm_in_plex_2d()

@pytest.mark.xfail
@pytest.mark.parallel(nprocs=2)
def test_pic_swarm_in_plex_3d():
    swarm = _test_pic_swarm_in_plex(UnitCubeMesh(1,1,1))

def _test_pic_swarm_remove_ghost_cell_coords(m):
    """Test that _test_pic_swarm_remove_ghost_cell_coords removes
    coordinates from ghost cells correctly."""
    m.init()
    pointcoords = cell_midpoints(m)
    plex = m.topology._plex
    swarm = mesh._pic_swarm_in_plex(plex, pointcoords)
    mesh._pic_swarm_remove_ghost_cell_coords(m, swarm)
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
    # (excluding ghost cells in the halo)
    assert len(localpointcoords) == m.cell_set.size
    # Check total number of points on all MPI ranks is correct
    # (excluding ghost cells in the halo)
    nptslocal = len(localpointcoords)
    nptsglobal = MPI.COMM_WORLD.allreduce(nptslocal, op=MPI.SUM)
    assert nptsglobal == len(pointcoords)
    assert nptsglobal == swarm.getSize()

@pytest.mark.xfail
def test_pic_swarm_remove_ghost_cell_coords_1d():
    _test_pic_swarm_remove_ghost_cell_coords(UnitIntervalMesh(1))

@pytest.mark.xfail
def test_pic_swarm_remove_ghost_cell_coords_2d():
    _test_pic_swarm_remove_ghost_cell_coords(UnitIntervalMesh(1,1))

@pytest.mark.xfail
def test_pic_swarm_remove_ghost_cell_coords_3d():
    _test_pic_swarm_remove_ghost_cell_coords(UnitIntervalMesh(1,1,1))

def verify_vertexonly_mesh(m, vm, gdim):
    # test that the mesh properties are as expected
    assert m.geometric_dimension() == gdim
    assert vm.geometric_dimension() == gdim
    assert vm.topological_dimension() == 0
    # Can initialise
    vm.init()
    # Check properties
    #TODO
    # Can create function spaces
    V = FunctionSpace(vm, "DG", 0)
    # Can't create function space other than DG0
    #TODO
    # Can create function on function spaces
    f = Function(V)
    # Can interpolate onto functions
    if gdim == 1:
        x, = SpatialCoordinate(vm)
        f.interpolate(x)
    elif gdim == 2:
        x, y = SpatialCoordinate(vm)
        f.interpolate(x+y)
    elif gdim == 3:
        x, y, z = SpatialCoordinate(vm)
        f.interpolate(x+y+z)
    # Get exact values at coordinates
    for coord in vm.coordinates.dat.data_ro:
        f.at(coord) == sum(coord)

def _test_generate(m):
    vertexcoords = cell_midpoints(m)
    vm = VertexOnlyMesh(m, vertexcoords)
    verify_vertexonly_mesh(m, vm, m.geometric_dimension())

@pytest.mark.xfail
def test_generate_1d():
    _test_generate(UnitIntervalMesh(1))

def test_generate_2d():
    _test_generate(UnitSquareMesh(1,1))

def test_generate_3d():
    _test_generate(UnitCubeMesh(1,1,1))

# remove this before final merge
if __name__ == "__main__":
    # f = Function(V).interpolate(m.coordinates)
    test_generate()
    import pytest, sys
    pytest.main([sys.argv[0]])
