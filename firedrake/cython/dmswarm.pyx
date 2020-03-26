# cython: language_level=3

# Utility functions to derive global and local numbering from DMSwarm
import cython
import numpy as np
from firedrake.petsc import PETSc
from mpi4py import MPI
from pyop2.datatypes import IntType
from libc.string cimport memset
from libc.stdlib cimport qsort

cimport numpy as np
cimport mpi4py.MPI as MPI
cimport petsc4py.PETSc as PETSc

np.import_array()

include "petschdr.pxi"

@cython.boundscheck(False)
@cython.wraparound(False)
def mark_entity_classes(PETSc.DM swarm):
    """Mark all points in a given DMSwarm according to the PyOP2 entity
    classes:

    core   : owned and not in send halo
    owned  : owned and in send halo
    ghost  : in halo

    Note that this is a copy and paste job from the equivalent function
    in dmplex.pyx with the relevant modifications made for a DMSwarm.

    :arg swarm: The DMSwarm object encapsulating the mesh topology
    """
    cdef:
        PetscInt pStart, pEnd, cStart, cEnd
        PetscInt c, ci, p
        PetscInt nleaves
        const PetscInt *ilocal = NULL
        PetscBool non_exec
        const PetscSFNode *iremote = NULL
        PETSc.SF point_sf = None
        PetscBool is_ghost, is_owned
        DMLabel lbl_core, lbl_owned, lbl_ghost

    pStart = 0
    pEnd = swarm.getSize()

    swarm.createLabel("pyop2_core")
    swarm.createLabel("pyop2_owned")
    swarm.createLabel("pyop2_ghost")

    CHKERR(DMGetLabel(swarm.dm, b"pyop2_core", &lbl_core))
    CHKERR(DMGetLabel(swarm.dm, b"pyop2_owned", &lbl_owned))
    CHKERR(DMGetLabel(swarm.dm, b"pyop2_ghost", &lbl_ghost))

    if swarm.comm.size > 1:
        # Mark ghosts from point overlap SF
        point_sf = swarm.getPointSF()
        CHKERR(PetscSFGetGraph(point_sf.sf, NULL, &nleaves, &ilocal, NULL))
        for p in range(nleaves):
            CHKERR(DMLabelSetValue(lbl_ghost, ilocal[p], 1))
    else:
        # If sequential mark all points as core
        for p in range(pStart, pEnd):
            CHKERR(DMLabelSetValue(lbl_core, p, 1))
        return

    CHKERR(DMLabelCreateIndex(lbl_ghost, pStart, pEnd))

    # Mark all remaining points as core
    CHKERR(DMLabelCreateIndex(lbl_owned, pStart, pEnd))
    for p in range(pStart, pEnd):
        CHKERR(DMLabelHasPoint(lbl_owned, p, &is_owned))
        CHKERR(DMLabelHasPoint(lbl_ghost, p, &is_ghost))
        if not is_ghost and not is_owned:
            CHKERR(DMLabelSetValue(lbl_core, p, 1))
    CHKERR(DMLabelDestroyIndex(lbl_owned))
    CHKERR(DMLabelDestroyIndex(lbl_ghost))

@cython.boundscheck(False)
@cython.wraparound(False)
def get_entity_classes(PETSc.DM swarm):
    """Builds PyOP2 entity class offsets for all entity levels.

    Note that this is a copy and paste job from the equivalent function
    in dmplex.pyx with the relevant modifications made for a DMSwarm.

    :arg swarm: The DMSwarm object encapsulating the mesh topology
    """
    cdef:
        np.ndarray[PetscInt, ndim=2, mode="c"] entity_class_sizes
        np.ndarray[PetscInt, mode="c"] eStart, eEnd
        PetscInt depth, d, i, ci, class_size, start, end
        const PetscInt *indices = NULL
        PETSc.IS class_is

    depth = 1 # by definition since a swarm is point cloud
    entity_class_sizes = np.zeros((depth, 3), dtype=IntType)
    eStart = np.zeros(depth, dtype=IntType)
    eEnd = np.zeros(depth, dtype=IntType)
    for d in range(depth):
        start = 0 # by definition since a swarm is point cloud
        CHKERR(DMSwarmGetSize(swarm.dm, &end)) # by definition since a swarm is point cloud
        eStart[d] = start
        eEnd[d] = end

    for i, op2class in enumerate([b"pyop2_core",
                                  b"pyop2_owned",
                                  b"pyop2_ghost"]):
        class_is = swarm.getStratumIS(op2class, 1)
        class_size = swarm.getStratumSize(op2class, 1)
        if class_size > 0:
            CHKERR(ISGetIndices(class_is.iset, &indices))
            for ci in range(class_size):
                for d in range(depth):
                    if eStart[d] <= indices[ci] < eEnd[d]:
                        entity_class_sizes[d, i] += 1
                        break
            CHKERR(ISRestoreIndices(class_is.iset, &indices))

    # PyOP2 entity class indices are additive
    for d in range(depth):
        for i in range(1, 3):
            entity_class_sizes[d, i] += entity_class_sizes[d, i-1]
    return entity_class_sizes

def create_section(mesh, nodes_per_entity, on_base=False):
    """Create the section describing a global numbering.

    Note that this is a copy and paste job from the equivalent function
    in dmplex.pyx with the relevant modifications made for a DMSwarm.

    :arg mesh: The mesh.
    :arg nodes_per_entity: Number of nodes on each type of topological
        entity of the mesh.
    :arg on_base: If True, assume extruded space is actually Foo x Real.

    :returns: A PETSc Section providing the number of dofs, and offset
        of each dof, on each mesh point.
    """
    # We don't use DMPlexCreateSection because we only ever put one
    # field in each section.
    cdef:
        PETSc.DM dm
        PETSc.Section section
        # PETSc.IS renumbering
        PetscInt i, p, layers, pStart, pEnd
        PetscInt dimension, ndof
        np.ndarray[PetscInt, ndim=2, mode="c"] nodes
        np.ndarray[PetscInt, ndim=2, mode="c"] layer_extents
        bint variable, extruded, on_base_

    variable = mesh.variable_layers
    extruded = mesh.cell_set._extruded
    on_base_ = on_base
    nodes_per_entity = np.asarray(nodes_per_entity, dtype=IntType)
    if variable:
        layer_extents = mesh.layer_extents
    elif extruded:
        if on_base:
            nodes_per_entity = sum(nodes_per_entity[:, i] for i in range(2))
        else:
            nodes_per_entity = sum(nodes_per_entity[:, i]*(mesh.layers - i) for i in range(2))

    dm = mesh._swarm
    # renumbering = mesh._swarm_renumbering
    section = PETSc.Section().create(comm=mesh.comm)
    pStart = 0 # by definition since point cloud
    pEnd = dm.getSize() # by definition since point cloud
    section.setChart(pStart, pEnd)
    # CHKERR(PetscSectionSetPermutation(section.sec, renumbering.iset))
    dimension = 0 # by definition since point cloud

    nodes = nodes_per_entity.reshape(dimension + 1, -1)

    for i in range(dimension + 1):
        pStart = 0 # by definition since a swarm is point cloud
        CHKERR(DMSwarmGetSize(dm.dm, &pEnd)) # by definition since a swarm is point cloud
        if not variable:
            ndof = nodes[i, 0]
        for p in range(pStart, pEnd):
            if variable:
                if on_base_:
                    ndof = nodes[i, 1]
                else:
                    layers = layer_extents[p, 1] - layer_extents[p, 0]
                    ndof = layers*nodes[i, 0] + (layers - 1)*nodes[i, 1]
            CHKERR(PetscSectionSetDof(section.sec, p, ndof))
    section.setUp()
    return section