import pytest
import numpy as np

from firedrake import *

def test_pic_swarm_in_plex():
    m = UnitSquareMesh(1,1)
    points = [(.1, .1), (.9, .9)]
    plex = m.topology._plex
    swarm = mesh._pic_swarm_in_plex(plex, points)
    # Now somehow check that points are in each cell as expected
    #TODO
    # Then check that distribution has worked across MPI ranks as expected
    #TODO

@pytest.mark.parallel(nprocs=2)
def test_pic_swarm_in_plex_parallel():
    test_pic_swarm_in_plex()

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
