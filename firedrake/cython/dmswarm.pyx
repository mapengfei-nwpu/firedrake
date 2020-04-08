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
import firedrake.cython.dmplex as dmplex

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

# TODO Complete below
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def closure_ordering(PETSc.DM plex,
#                      PETSc.Section vertex_numbering,
#                      PETSc.Section cell_numbering,
#                      np.ndarray[PetscInt, ndim=1, mode="c"] entity_per_cell):
#     """Apply Fenics local numbering to a cell closure.

#     :arg plex: The DMSwarm object encapsulating the vertex-only mesh topology
#     :arg vertex_numbering: Section describing the universal vertex numbering
#     :arg cell_numbering: Section describing the global cell numbering
#     :arg entity_per_cell: List of the number of entity points in each dimension

#     Vertices    := Ordered according to global/universal
#                    vertex numbering
#     Edges/faces := Ordered according to lexicographical
#                    ordering of non-incident vertices
#     """
#     cdef:
#         PetscInt c, cStart, cEnd, v, vStart, vEnd
#         PetscInt f, fStart, fEnd, e, eStart, eEnd
#         PetscInt dim, vi, ci, fi, v_per_cell, cell
#         PetscInt offset, cell_offset, nfaces, nfacets
#         PetscInt nclosure, nfacet_closure, nface_vertices
#         PetscInt *vertices = NULL
#         PetscInt *v_global = NULL
#         PetscInt *closure = NULL
#         PetscInt *facets = NULL
#         PetscInt *faces = NULL
#         PetscInt *face_indices = NULL
#         const PetscInt *face_vertices = NULL
#         PetscInt *facet_vertices = NULL
#         np.ndarray[PetscInt, ndim=2, mode="c"] cell_closure

#     dim = plex.getDimension()
#     cStart, cEnd = plex.getHeightStratum(0)
#     fStart, fEnd = plex.getHeightStratum(1)
#     eStart, eEnd = plex.getDepthStratum(1)
#     vStart, vEnd = plex.getDepthStratum(0)
#     v_per_cell = entity_per_cell[0]
#     cell_offset = sum(entity_per_cell) - 1

#     CHKERR(PetscMalloc1(v_per_cell, &vertices))
#     CHKERR(PetscMalloc1(v_per_cell, &v_global))
#     CHKERR(PetscMalloc1(v_per_cell, &facets))
#     CHKERR(PetscMalloc1(v_per_cell-1, &facet_vertices))
#     CHKERR(PetscMalloc1(entity_per_cell[1], &faces))
#     CHKERR(PetscMalloc1(entity_per_cell[1], &face_indices))
#     cell_closure = np.empty((cEnd - cStart, sum(entity_per_cell)), dtype=IntType)

#     for c in range(cStart, cEnd):
#         CHKERR(PetscSectionGetOffset(cell_numbering.sec, c, &cell))
#         CHKERR(DMPlexGetTransitiveClosure(plex.dm, c, PETSC_TRUE,
#                                           &nclosure,&closure))

#         # Find vertices and translate universal numbers
#         vi = 0
#         for ci in range(nclosure):
#             if vStart <= closure[2*ci] < vEnd:
#                 vertices[vi] = closure[2*ci]
#                 CHKERR(PetscSectionGetOffset(vertex_numbering.sec,
#                                              closure[2*ci], &v))
#                 # Correct -ve offsets for non-owned entities
#                 if v >= 0:
#                     v_global[vi] = v
#                 else:
#                     v_global[vi] = -(v+1)
#                 vi += 1

#         # Sort vertices by universal number
#         CHKERR(PetscSortIntWithArray(v_per_cell,v_global,vertices))
#         for vi in range(v_per_cell):
#             if dim == 1:
#                 # Correct 1D edge numbering
#                 cell_closure[cell, vi] = vertices[v_per_cell-vi-1]
#             else:
#                 cell_closure[cell, vi] = vertices[vi]
#         offset = v_per_cell

#         # Find all edges (dim=1)
#         if dim > 2:
#             nfaces = 0
#             for ci in range(nclosure):
#                 if eStart <= closure[2*ci] < eEnd:
#                     faces[nfaces] = closure[2*ci]

#                     CHKERR(DMPlexGetConeSize(plex.dm, closure[2*ci],
#                                              &nface_vertices))
#                     CHKERR(DMPlexGetCone(plex.dm, closure[2*ci],
#                                          &face_vertices))

#                     # Edges in 3D are tricky because we need a
#                     # lexicographical sort with two keys (the local
#                     # numbers of the two non-incident vertices).

#                     # Find non-incident vertices
#                     fi = 0
#                     face_indices[nfaces] = 0
#                     for v in range(v_per_cell):
#                         incident = 0
#                         for vi in range(nface_vertices):
#                             if cell_closure[cell,v] == face_vertices[vi]:
#                                 incident = 1
#                                 break
#                         if incident == 0:
#                             face_indices[nfaces] += v * 10**(1-fi)
#                             fi += 1
#                     nfaces += 1

#             # Sort by local numbers of non-incident vertices
#             CHKERR(PetscSortIntWithArray(entity_per_cell[1],
#                                          face_indices, faces))
#             for fi in range(nfaces):
#                 cell_closure[cell, offset+fi] = faces[fi]
#             offset += nfaces

#         # Calling DMPlexGetTransitiveClosure() again invalidates the
#         # current work array, so we need to get the facets and cell
#         # out before getting the facet closures.

#         # Find all facets (co-dim=1)
#         nfacets = 0
#         for ci in range(nclosure):
#             if fStart <= closure[2*ci] < fEnd:
#                 facets[nfacets] = closure[2*ci]
#                 nfacets += 1

#         # The cell itself is always the first entry in the Plex closure
#         cell_closure[cell, cell_offset] = closure[0]

#         # Now we can deal with facets
#         if dim > 1:
#             for f in range(nfacets):
#                 # Derive facet vertices from facet_closure
#                 CHKERR(DMPlexGetTransitiveClosure(plex.dm, facets[f],
#                                                   PETSC_TRUE,
#                                                   &nfacet_closure,
#                                                   &closure))
#                 vi = 0
#                 for fi in range(nfacet_closure):
#                     if vStart <= closure[2*fi] < vEnd:
#                         facet_vertices[vi] = closure[2*fi]
#                         vi += 1

#                 # Find non-incident vertices
#                 for v in range(v_per_cell):
#                     incident = 0
#                     for vi in range(v_per_cell-1):
#                         if cell_closure[cell,v] == facet_vertices[vi]:
#                             incident = 1
#                             break
#                     # Only one non-incident vertex per facet, so
#                     # local facet no. = non-incident vertex no.
#                     if incident == 0:
#                         cell_closure[cell,offset+v] = facets[f]
#                         break

#             offset += nfacets

#     if closure != NULL:
#         CHKERR(DMPlexRestoreTransitiveClosure(plex.dm, 0, PETSC_TRUE,
#                                               NULL, &closure))
#     CHKERR(PetscFree(vertices))
#     CHKERR(PetscFree(v_global))
#     CHKERR(PetscFree(facets))
#     CHKERR(PetscFree(facet_vertices))
#     CHKERR(PetscFree(faces))
#     CHKERR(PetscFree(face_indices))

#     return cell_closure

get_cell_nodes = dmplex.get_cell_nodes