#ifndef AMR_SHIM_H
#define AMR_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AmrForest AmrForest;

/* Refine criterion. Receives the cell center (cx, cy, cz) and cell edge
 * length h, both in the user's physical units (defined by the domain
 * bounds passed to amr_forest_new). Return non-zero to subdivide.
 * ctx is a user pointer passed through unchanged. */
typedef int (*AmrRefineFn)(double cx, double cy, double cz, double h, void *ctx);

/* One-time process init / shutdown. Internally calls MPI_Init,
 * sc_init, p4est_init. Safe to call once per process only. */
int  amr_init(void);
void amr_finalize(void);

/* Build a forest of brick-connected unit-cube trees mapped onto the
 * physical box [xmin,xmax] × [ymin,ymax] × [zmin,zmax]. The brick has
 * trees_x × trees_y × trees_z root trees; each tree is a cube in the
 * forest's logical space and the physical map is a single affine
 * transform (scale + offset). For the resulting cells to be isotropic
 * in physical space, choose trees_{x,y,z} so the physical extent per
 * tree is the same in all 3 directions.
 *
 * min_level forces uniform refinement to that depth. Returns NULL on
 * failure. */
AmrForest *amr_forest_new(int trees_x, int trees_y, int trees_z,
                          double xmin, double xmax,
                          double ymin, double ymax,
                          double zmin, double zmax,
                          int min_level);

void amr_forest_destroy(AmrForest *f);

/* Recursively refine each leaf for which fn returns non-zero, capped at
 * max_level, then 2:1-balance and partition. */
void amr_forest_refine(AmrForest *f, int max_level, AmrRefineFn fn, void *ctx);

/* Coarsen criterion. Receives the *parent* center / edge length the 8
 * children would merge into. Return non-zero to merge. ctx is a user
 * pointer passed through unchanged. */
typedef int (*AmrCoarsenFn)(double cx, double cy, double cz, double h, void *ctx);

/* Recursively coarsen 8-sibling families for which fn returns non-zero,
 * stopping at min_level (so the resulting cell stays at least that
 * coarse), then 2:1-balance and partition. */
void amr_forest_coarsen(AmrForest *f, int min_level, AmrCoarsenFn fn, void *ctx);

/* Write {filename}.vtu (and pvtu in parallel) showing the leaf cells. */
void amr_forest_write_vtk(AmrForest *f, const char *filename);

/* Total leaf count across all ranks. */
int64_t amr_forest_leaf_count(const AmrForest *f);

/* Leaf count on the calling rank only (== local_num_quadrants). */
int64_t amr_forest_n_local_leaves(const AmrForest *f);

/* The cell edge length in physical units at refinement level L, given
 * the forest's brick + domain. Useful for comparing to a uniform grid. */
double amr_forest_cell_size_at_level(const AmrForest *f, int level);

/* Per-leaf metadata in physical units. Returned in p4est's Morton-ordered
 * traversal; the index of a leaf in this array is its stable flat ID
 * within the rank between refines (it changes after a regrid). */
typedef struct AmrLeafInfo {
    int32_t  tree_id;     /* which root tree this leaf is in */
    int8_t   level;       /* refinement level (0 = whole tree) */
    int8_t   _pad[7];
    double   center[3];   /* cell center in user physical units */
    double   size[3];     /* cell edge lengths in user physical units */
} AmrLeafInfo;

/* Fill `out` (length n_local_leaves) with the local leaves in Morton
 * order. The caller owns the buffer. */
void amr_forest_fill_leaves(const AmrForest *f, AmrLeafInfo *out);

/* ─── Phase 3a: face-neighbor queries via p8est_mesh ─────────────────── */

/* Result of a single face-neighbor query.
 *
 * For a 2:1-balanced forest, a leaf's face neighbor falls into exactly
 * one of these cases:
 *   AMR_NB_BOUNDARY   — physical domain boundary, no neighbor.
 *   AMR_NB_SAME       — single neighbor at the same refinement level.
 *   AMR_NB_COARSER    — single neighbor one level COARSER than this leaf.
 *                       This leaf is on a hanging face of the coarser one;
 *                       three sibling leaves (this + 3 same-level peers)
 *                       all touch the same coarse face.
 *   AMR_NB_FINER      — four neighbors one level FINER than this leaf.
 *                       This leaf has the hanging face; the four fine
 *                       leaves cover this leaf's face. count == 4.
 *   AMR_NB_GHOST      — neighbor lives on another MPI rank. count == 1
 *                       and idx[0] is the ghost-layer index, not a local
 *                       leaf index. (Phase 7 surfaces these properly.)
 */
typedef enum {
    AMR_NB_BOUNDARY = 0,
    AMR_NB_SAME     = 1,
    AMR_NB_COARSER  = 2,
    AMR_NB_FINER    = 3,
    AMR_NB_GHOST    = 4,
} AmrNeighborKind;

typedef struct AmrFaceNeighbors {
    int32_t kind;       /* AmrNeighborKind */
    int32_t count;      /* 0 (BOUNDARY), 1 (SAME/COARSER/GHOST), or 4 (FINER) */
    int64_t idx[4];     /* local leaf indices for SAME/COARSER/FINER local slots;
                         * ghost-layer index for GHOST and for FINER ghost slots
                         * (see ghost_mask). -1 for unused slots. */
    int32_t ghost_mask; /* FINER only: bit h set ⇒ idx[h] is a ghost-layer index
                         * (a finer neighbor on another rank) rather than a local
                         * leaf index. 0 for all other kinds. */
} AmrFaceNeighbors;

/* Get the face neighbors of leaf `leaf_idx` across face `face` (0..5
 * = x-, x+, y-, y+, z-, z+). Mesh must be available; this is true after
 * any call to amr_forest_refine. Writes into `*out`. */
void amr_forest_face_neighbors(const AmrForest *f,
                               int64_t leaf_idx,
                               int     face,
                               AmrFaceNeighbors *out);

/* ─── Phase 3a: spatial point query via p8est_search_local ───────────── */

/* Find the local leaf containing the point (x, y, z) in physical units.
 * Returns the leaf flat index in [0, n_local_leaves), or -1 if the
 * point is outside the rank-local subdomain (or outside the global
 * domain). */
int64_t amr_forest_search_point(const AmrForest *f, double x, double y, double z);

/* Query p8est's global partition directory (global_first_position). These
 * calls are local: every rank already carries the compact partition markers.
 * Point ownership uses the same half-open convention as p8est, with the
 * physical high domain boundary included. */
int amr_forest_owner_rank(const AmrForest *f, double x, double y, double z);

/* Mark every owner rank whose partition has positive-volume overlap with the
 * physical axis-aligned box. `out` has mpisize entries and is zeroed here.
 * Degenerate boxes are point queries. Returns the number of marked ranks. */
int amr_forest_overlapping_ranks(const AmrForest *f,
                                 const double lo[3], const double hi[3],
                                 int32_t *out);

/* ─── Phase 7: cross-rank ghost layer exposure ───────────────────────────
 * Accessors over the p4est ghost layer for cross-rank field exchange. On a
 * single rank the ghost layer is empty, so every count below is 0. */

int     amr_forest_mpisize(const AmrForest *f);
int     amr_forest_mpirank(const AmrForest *f);

/* Number of ghost (received-from-other-ranks) quadrants. */
int64_t amr_forest_ghost_count(const AmrForest *f);

/* Geometry of each ghost quadrant (reuses AmrLeafInfo), ghost-layer order. */
void    amr_forest_fill_ghosts(const AmrForest *f, AmrLeafInfo *out);

/* Ghost recv ranges by owner rank; `out` has mpisize+1 entries. */
void    amr_forest_ghost_proc_offsets(const AmrForest *f, int32_t *out);

/* Total mirror-send slots (sum over ranks of local cells ghosted elsewhere). */
int64_t amr_forest_mirror_send_count(const AmrForest *f);

/* Mirror send ranges by target rank; `out` has mpisize+1 entries. */
void    amr_forest_mirror_proc_offsets(const AmrForest *f, int32_t *out);

/* Local quadrant index of each mirror-send slot; `out` has
 * amr_forest_mirror_send_count() entries, ordered to match the receiver. */
void    amr_forest_mirror_locals(const AmrForest *f, int32_t *out);

#ifdef __cplusplus
}
#endif

#endif /* AMR_SHIM_H */
