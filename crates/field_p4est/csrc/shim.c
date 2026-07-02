#include "shim.h"

#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include <p8est_bits.h>
#include <p8est_extended.h>
#include <p8est_ghost.h>
#include <p8est_mesh.h>
#include <p8est_search.h>
#include <p8est_vtk.h>

struct AmrForest {
  p8est_connectivity_t *conn;
  p8est_t              *p8est;
  p8est_ghost_t        *ghost;   /* (re)built after every refine */
  p8est_mesh_t         *mesh;    /* (re)built after every refine */
  /* Physical-domain affine map: phys = origin + scale * (logical brick coord).
   * For a brick of (Tx, Ty, Tz) unit-cube trees mapped onto
   * [xmin,xmax]×[ymin,ymax]×[zmin,zmax], the brick's logical coord runs
   * [0,Tx]×[0,Ty]×[0,Tz] and scale = (Lx/Tx, Ly/Ty, Lz/Tz). */
  double origin[3];
  double scale[3];
  int    trees[3];
};

static void rebuild_mesh(AmrForest *f) {
  if (f->mesh)  { p8est_mesh_destroy(f->mesh);  f->mesh  = NULL; }
  if (f->ghost) { p8est_ghost_destroy(f->ghost); f->ghost = NULL; }
  f->ghost = p8est_ghost_new(f->p8est, P8EST_CONNECT_FULL);
  f->mesh  = p8est_mesh_new(f->p8est, f->ghost, P8EST_CONNECT_FULL);
}

typedef struct {
  AmrRefineFn  fn;
  void        *ctx;
  int          max_level;
  AmrForest   *forest;
} refine_ctx_t;

static int amr_initialized = 0;
static int amr_owns_mpi = 0;

int amr_init(void) {
  if (amr_initialized) return 0;
  // Cooperate with hosts that already initialized MPI themselves
  // (e.g., grass_mpi via rsmpi). Skip sc_MPI_Init in that case so we
  // don't trigger Open MPI's "MPI_INIT called more than once" abort.
  int mpi_already = 0;
  MPI_Initialized(&mpi_already);
  if (!mpi_already) {
    int mpiret = sc_MPI_Init(NULL, NULL);
    if (mpiret != sc_MPI_SUCCESS) return -1;
    amr_owns_mpi = 1;
  } else {
    amr_owns_mpi = 0;
  }
  sc_init(sc_MPI_COMM_WORLD, 1, 1, NULL, SC_LP_ERROR);
  p4est_init(NULL, SC_LP_ERROR);
  amr_initialized = 1;
  return 0;
}

void amr_finalize(void) {
  if (!amr_initialized) return;
  sc_finalize();
  // Only finalize MPI if we initialized it ourselves; otherwise the
  // host owns the lifecycle.
  if (amr_owns_mpi) {
    MPI_Finalize();
    amr_owns_mpi = 0;
  }
  amr_initialized = 0;
}

AmrForest *amr_forest_new(int trees_x, int trees_y, int trees_z,
                          double xmin, double xmax,
                          double ymin, double ymax,
                          double zmin, double zmax,
                          int min_level) {
  AmrForest *f = (AmrForest *) calloc(1, sizeof(AmrForest));

  f->conn = p8est_connectivity_new_brick(trees_x, trees_y, trees_z, 0, 0, 0);

  f->origin[0] = xmin;
  f->origin[1] = ymin;
  f->origin[2] = zmin;
  f->scale[0]  = (xmax - xmin) / (double) trees_x;
  f->scale[1]  = (ymax - ymin) / (double) trees_y;
  f->scale[2]  = (zmax - zmin) / (double) trees_z;
  f->trees[0]  = trees_x;
  f->trees[1]  = trees_y;
  f->trees[2]  = trees_z;

  f->p8est = p8est_new_ext(sc_MPI_COMM_WORLD,
                           f->conn,
                           /* min_quadrants */ 0,
                           /* min_level     */ min_level,
                           /* fill_uniform  */ 1,
                           /* data_size     */ 0,
                           /* init_fn       */ NULL,
                           /* user_pointer  */ NULL);
  rebuild_mesh(f);
  return f;
}

void amr_forest_destroy(AmrForest *f) {
  if (!f) return;
  if (f->mesh)  p8est_mesh_destroy(f->mesh);
  if (f->ghost) p8est_ghost_destroy(f->ghost);
  if (f->p8est) p8est_destroy(f->p8est);
  if (f->conn)  p8est_connectivity_destroy(f->conn);
  free(f);
}

static int refine_trampoline(p8est_t *p8est,
                             p4est_topidx_t which_tree,
                             p8est_quadrant_t *q) {
  refine_ctx_t *rctx = (refine_ctx_t *) p8est->user_pointer;
  AmrForest    *f    = rctx->forest;

  if ((int) q->level >= rctx->max_level) return 0;

  /* Use p8est_qcoord_to_vertex to get the cell's two opposing corners in
   * the brick's logical coord system [0,Tx]×[0,Ty]×[0,Tz]. */
  double v0[3], v1[3];
  p4est_qcoord_t qlen = P8EST_QUADRANT_LEN(q->level);
  p8est_qcoord_to_vertex(f->conn, which_tree, q->x,        q->y,        q->z,        v0);
  p8est_qcoord_to_vertex(f->conn, which_tree, q->x + qlen, q->y + qlen, q->z + qlen, v1);

  /* Center and edge length in physical units. The brick map is uniform
   * per axis (scale[i] is the per-tree physical extent), so a cube in
   * logical space stays a cube in physical space iff scale[0]==scale[1]==scale[2]. */
  double cx = f->origin[0] + 0.5 * (v0[0] + v1[0]) * f->scale[0];
  double cy = f->origin[1] + 0.5 * (v0[1] + v1[1]) * f->scale[1];
  double cz = f->origin[2] + 0.5 * (v0[2] + v1[2]) * f->scale[2];
  double hx = (v1[0] - v0[0]) * f->scale[0];
  double hy = (v1[1] - v0[1]) * f->scale[1];
  double hz = (v1[2] - v0[2]) * f->scale[2];

  /* Conservative for isotropic distance criteria: pass the max edge. */
  double h = hx;
  if (hy > h) h = hy;
  if (hz > h) h = hz;

  return rctx->fn(cx, cy, cz, h, rctx->ctx);
}

void amr_forest_refine(AmrForest *f, int max_level, AmrRefineFn fn, void *ctx) {
  refine_ctx_t rctx = { fn, ctx, max_level, f };
  void *prev_user = f->p8est->user_pointer;
  f->p8est->user_pointer = &rctx;

  p8est_refine(f->p8est, /* recursive */ 1, refine_trampoline, NULL);
  p8est_balance(f->p8est, P8EST_CONNECT_FULL, NULL);
  p8est_partition(f->p8est, /* allow_for_coarsening */ 0, NULL);

  f->p8est->user_pointer = prev_user;
  rebuild_mesh(f);
}

/* ─── Phase 6: coarsen / adapt ─────────────────────────────────────────
 *
 * Coarsen merges 8 sibling fine quadrants back into their common parent
 * when `fn(cx, cy, cz, h, ctx)` returns non-zero on each of the 8
 * children (we evaluate on the BARYCENTER of the 8 — siblings share a
 * parent and have identical "should-coarsen" intent in our callback
 * model, so we just check the parent's center).
 */

typedef struct {
  AmrCoarsenFn fn;
  void        *ctx;
  int          min_level;
  AmrForest   *forest;
} coarsen_ctx_t;

static int coarsen_trampoline(p8est_t *p8est,
                              p4est_topidx_t which_tree,
                              p8est_quadrant_t *children[]) {
  coarsen_ctx_t *cctx = (coarsen_ctx_t *) p8est->user_pointer;
  AmrForest     *f    = cctx->forest;

  /* Don't coarsen below `min_level`. children[0] is one of 8 siblings
   * — their level is one finer than the parent, so the parent would
   * be at children[0]->level - 1. */
  int parent_level = (int) children[0]->level - 1;
  if (parent_level < cctx->min_level) return 0;

  /* Parent center: average of any sibling pair across the parent's
   * extent. Take child 0 (the lowest-corner sibling) and project to
   * its parent's bbox. The parent has 2x the side length and starts
   * at the same lo corner as children[0]. */
  p4est_qcoord_t plen = P8EST_QUADRANT_LEN(parent_level);
  double v0[3], v1[3];
  p8est_qcoord_to_vertex(f->conn, which_tree,
                         children[0]->x, children[0]->y, children[0]->z, v0);
  p8est_qcoord_to_vertex(f->conn, which_tree,
                         children[0]->x + plen, children[0]->y + plen,
                         children[0]->z + plen, v1);
  double cx = f->origin[0] + 0.5 * (v0[0] + v1[0]) * f->scale[0];
  double cy = f->origin[1] + 0.5 * (v0[1] + v1[1]) * f->scale[1];
  double cz = f->origin[2] + 0.5 * (v0[2] + v1[2]) * f->scale[2];
  double h  = (v1[0] - v0[0]) * f->scale[0];
  /* Use max axis if scales differ. */
  double hy = (v1[1] - v0[1]) * f->scale[1]; if (hy > h) h = hy;
  double hz = (v1[2] - v0[2]) * f->scale[2]; if (hz > h) h = hz;

  return cctx->fn(cx, cy, cz, h, cctx->ctx);
}

void amr_forest_coarsen(AmrForest *f, int min_level, AmrCoarsenFn fn, void *ctx) {
  coarsen_ctx_t cctx = { fn, ctx, min_level, f };
  void *prev_user = f->p8est->user_pointer;
  f->p8est->user_pointer = &cctx;

  p8est_coarsen(f->p8est, /* recursive */ 1, coarsen_trampoline, NULL);
  p8est_balance(f->p8est, P8EST_CONNECT_FULL, NULL);
  p8est_partition(f->p8est, /* allow_for_coarsening */ 0, NULL);

  f->p8est->user_pointer = prev_user;
  rebuild_mesh(f);
}

void amr_forest_write_vtk(AmrForest *f, const char *filename) {
  p8est_vtk_write_file(f->p8est, NULL, filename);
}

int64_t amr_forest_leaf_count(const AmrForest *f) {
  return (int64_t) f->p8est->global_num_quadrants;
}

int64_t amr_forest_n_local_leaves(const AmrForest *f) {
  return (int64_t) f->p8est->local_num_quadrants;
}

void amr_forest_fill_leaves(const AmrForest *f, AmrLeafInfo *out) {
  /* Walk every local tree's flat quadrant array. p8est stores leaves
   * tree-by-tree in Morton order within each tree; concatenating them
   * gives the rank-local Morton order. */
  size_t out_idx = 0;
  for (p4est_topidx_t t = f->p8est->first_local_tree;
       t <= f->p8est->last_local_tree;
       ++t) {
    p8est_tree_t *tree = p8est_tree_array_index(f->p8est->trees, t);
    sc_array_t   *quads = &tree->quadrants;
    size_t        nq = quads->elem_count;
    for (size_t q = 0; q < nq; ++q) {
      p8est_quadrant_t *quadrant = p8est_quadrant_array_index(quads, q);
      p4est_qcoord_t    qlen = P8EST_QUADRANT_LEN(quadrant->level);

      double v0[3], v1[3];
      p8est_qcoord_to_vertex(f->conn, t,
                             quadrant->x,        quadrant->y,        quadrant->z,        v0);
      p8est_qcoord_to_vertex(f->conn, t,
                             quadrant->x + qlen, quadrant->y + qlen, quadrant->z + qlen, v1);

      AmrLeafInfo *info = &out[out_idx++];
      info->tree_id   = (int32_t) t;
      info->level     = (int8_t)  quadrant->level;
      info->center[0] = f->origin[0] + 0.5 * (v0[0] + v1[0]) * f->scale[0];
      info->center[1] = f->origin[1] + 0.5 * (v0[1] + v1[1]) * f->scale[1];
      info->center[2] = f->origin[2] + 0.5 * (v0[2] + v1[2]) * f->scale[2];
      info->size[0]   = (v1[0] - v0[0]) * f->scale[0];
      info->size[1]   = (v1[1] - v0[1]) * f->scale[1];
      info->size[2]   = (v1[2] - v0[2]) * f->scale[2];
    }
  }
}

double amr_forest_cell_size_at_level(const AmrForest *f, int level) {
  /* Returns the smaller of the per-axis cell sizes at the given level.
   * For an isotropic brick (scale[*] all equal), all axes agree. */
  double s = f->scale[0];
  if (f->scale[1] < s) s = f->scale[1];
  if (f->scale[2] < s) s = f->scale[2];
  return s / (double) (1 << level);
}

/* ─── Phase 3a: face neighbors via p8est_mesh ────────────────────────── */

void amr_forest_face_neighbors(const AmrForest *f,
                               int64_t leaf_idx,
                               int     face,
                               AmrFaceNeighbors *out) {
  out->kind  = AMR_NB_BOUNDARY;
  out->count = 0;
  out->idx[0] = -1; out->idx[1] = -1; out->idx[2] = -1; out->idx[3] = -1;
  out->ghost_mask = 0;

  p8est_mesh_t  *mesh  = f->mesh;
  p4est_locidx_t local = mesh->local_num_quadrants;

  /* Guard the i64->locidx(i32) narrowing: a leaf_idx past the local count (or
   * one that would overflow the 6*leaf_idx index) is out of contract; return
   * BOUNDARY rather than read out of bounds in quad_to_quad/quad_to_face. */
  if (leaf_idx < 0 || leaf_idx >= (int64_t) local || face < 0 || face >= 6) {
    return;
  }
  p4est_locidx_t entry = (p4est_locidx_t)(6 * leaf_idx + face);

  p4est_locidx_t qq = mesh->quad_to_quad[entry];
  int8_t         qf = mesh->quad_to_face[entry];

  /* Boundary: a leaf on the domain boundary "sees itself" with the same
   * face index, per the p8est_mesh docs. */
  if (qq == (p4est_locidx_t) leaf_idx && qf == (int8_t) face) {
    return;
  }

  if (qf >= 0 && qf < 24) {
    /* Same-size neighbor. */
    if (qq < local) {
      out->kind  = AMR_NB_SAME;
      out->count = 1;
      out->idx[0] = (int64_t) qq;
    } else {
      out->kind  = AMR_NB_GHOST;
      out->count = 1;
      out->idx[0] = (int64_t)(qq - local);
    }
  } else if (qf >= 24 && qf < 120) {
    /* Double-size neighbor: this leaf is on a hanging face of a coarser
     * neighbor. Single neighbor, idx points at the coarser cell. */
    if (qq < local) {
      out->kind  = AMR_NB_COARSER;
      out->count = 1;
      out->idx[0] = (int64_t) qq;
    } else {
      out->kind  = AMR_NB_GHOST;
      out->count = 1;
      out->idx[0] = (int64_t)(qq - local);
    }
  } else if (qf >= -24 && qf < 0) {
    /* Half-size: four finer neighbors. qq indexes into quad_to_half,
     * which is an sc_array of struct { p4est_locidx_t [4] }. Each half may be
     * local (idx < local) or a ghost on another rank (idx >= local); we report
     * the split per slot via ghost_mask so the caller can emit the cross-rank
     * coarse/fine sub-faces. */
    p4est_locidx_t *halves = (p4est_locidx_t *)
      sc_array_index_int(mesh->quad_to_half, (int) qq);
    out->kind  = AMR_NB_FINER;
    out->count = 4;
    for (int h = 0; h < 4; ++h) {
      if (halves[h] >= local) {
        out->idx[h]      = (int64_t)(halves[h] - local); /* ghost-layer index */
        out->ghost_mask |= (1 << h);
      } else {
        out->idx[h] = (int64_t) halves[h];               /* local leaf index */
      }
    }
  }
}

/* ─── Phase 3a: spatial point query via p8est_search_local ──────────── */

typedef struct {
  double            x, y, z;
  int64_t           result;
  const AmrForest  *forest;
} search_ctx_t;

/* Test: is the physical point inside this quadrant's bbox?
 * If yes, return 1 (descend / claim leaf); if no, return 0. */
static int search_point_qf(p8est_t *p4est,
                           p4est_topidx_t which_tree,
                           p8est_quadrant_t *q,
                           p4est_locidx_t local_num,
                           void *point) {
  (void) point;
  search_ctx_t *ctx = (search_ctx_t *) p4est->user_pointer;
  const AmrForest *f = ctx->forest;

  double v0[3], v1[3];
  p4est_qcoord_t qlen = P8EST_QUADRANT_LEN(q->level);
  p8est_qcoord_to_vertex(f->conn, which_tree, q->x,        q->y,        q->z,        v0);
  p8est_qcoord_to_vertex(f->conn, which_tree, q->x + qlen, q->y + qlen, q->z + qlen, v1);

  double x0 = f->origin[0] + v0[0] * f->scale[0];
  double y0 = f->origin[1] + v0[1] * f->scale[1];
  double z0 = f->origin[2] + v0[2] * f->scale[2];
  double x1 = f->origin[0] + v1[0] * f->scale[0];
  double y1 = f->origin[1] + v1[1] * f->scale[1];
  double z1 = f->origin[2] + v1[2] * f->scale[2];

  /* Half-open at the high side so a point on a face goes to the lower
   * cell, avoiding double-claims at boundaries. */
  if (ctx->x <  x0 || ctx->x >= x1 ||
      ctx->y <  y0 || ctx->y >= y1 ||
      ctx->z <  z0 || ctx->z >= z1) {
    return 0;
  }
  if (local_num >= 0) {
    ctx->result = (int64_t) local_num;
  }
  return 1;
}

int64_t amr_forest_search_point(const AmrForest *f, double x, double y, double z) {
  search_ctx_t ctx = { x, y, z, -1, f };
  void *prev = f->p8est->user_pointer;
  f->p8est->user_pointer = &ctx;
  p8est_search_local(f->p8est,
                     /* call_post */ 0,
                     search_point_qf,
                     /* point_fn  */ NULL,
                     /* points    */ NULL);
  f->p8est->user_pointer = prev;
  return ctx.result;
}

/* ─── Phase 7: cross-rank ghost layer exposure ───────────────────────────
 *
 * p4est's ghost layer already underlies the AMR_NB_GHOST neighbor results.
 * These accessors expose it for field exchange: the (received) ghost
 * quadrants' geometry + owner-rank ranges, and the local "mirror" quadrants
 * other ranks need. p4est guarantees the mirrors this rank sends to rank R
 * are in the SAME order as the ghosts rank R receives from this rank, so a
 * matched send/recv halo plan lines up by construction. */

int amr_forest_mpisize(const AmrForest *f) { return f->p8est->mpisize; }
int amr_forest_mpirank(const AmrForest *f) { return f->p8est->mpirank; }

int64_t amr_forest_ghost_count(const AmrForest *f) {
  return (int64_t) f->ghost->ghosts.elem_count;
}

/* Geometry of every ghost (received) quadrant, in ghost-layer order
 * (grouped by owner rank ascending — matching ghost_proc_offsets). */
void amr_forest_fill_ghosts(const AmrForest *f, AmrLeafInfo *out) {
  p8est_ghost_t  *g = f->ghost;
  size_t          n = g->ghosts.elem_count;
  p4est_topidx_t  num_trees = (p4est_topidx_t) f->conn->num_trees;
  p4est_topidx_t  t = 0;
  for (size_t i = 0; i < n; ++i) {
    while (t + 1 < num_trees && (size_t) g->tree_offsets[t + 1] <= i) ++t;
    p8est_quadrant_t *q = p8est_quadrant_array_index(&g->ghosts, i);
    p4est_qcoord_t    qlen = P8EST_QUADRANT_LEN(q->level);
    double v0[3], v1[3];
    p8est_qcoord_to_vertex(f->conn, t, q->x,        q->y,        q->z,        v0);
    p8est_qcoord_to_vertex(f->conn, t, q->x + qlen, q->y + qlen, q->z + qlen, v1);
    AmrLeafInfo *info = &out[i];
    info->tree_id   = (int32_t) t;
    info->level     = (int8_t)  q->level;
    info->center[0] = f->origin[0] + 0.5 * (v0[0] + v1[0]) * f->scale[0];
    info->center[1] = f->origin[1] + 0.5 * (v0[1] + v1[1]) * f->scale[1];
    info->center[2] = f->origin[2] + 0.5 * (v0[2] + v1[2]) * f->scale[2];
    info->size[0]   = (v1[0] - v0[0]) * f->scale[0];
    info->size[1]   = (v1[1] - v0[1]) * f->scale[1];
    info->size[2]   = (v1[2] - v0[2]) * f->scale[2];
  }
}

/* Ghost recv ranges by owner rank: out[0..mpisize] — ghosts [out[r],out[r+1])
 * are owned by rank r. */
void amr_forest_ghost_proc_offsets(const AmrForest *f, int32_t *out) {
  p8est_ghost_t *g = f->ghost;
  int mpisize = f->p8est->mpisize;
  for (int r = 0; r <= mpisize; ++r) out[r] = (int32_t) g->proc_offsets[r];
}

/* Total mirror-send slots across all ranks. */
int64_t amr_forest_mirror_send_count(const AmrForest *f) {
  int mpisize = f->p8est->mpisize;
  return (int64_t) f->ghost->mirror_proc_offsets[mpisize];
}

/* Mirror send ranges by target rank: out[0..mpisize]. */
void amr_forest_mirror_proc_offsets(const AmrForest *f, int32_t *out) {
  p8est_ghost_t *g = f->ghost;
  int mpisize = f->p8est->mpisize;
  for (int r = 0; r <= mpisize; ++r) out[r] = (int32_t) g->mirror_proc_offsets[r];
}

/* Local quadrant index of each mirror-send slot k in [0, mirror_send_count).
 * Ordered by mirror_proc_offsets, matching the receiver's ghost order. */
void amr_forest_mirror_locals(const AmrForest *f, int32_t *out) {
  p8est_ghost_t *g = f->ghost;
  int mpisize = f->p8est->mpisize;
  int64_t total = (int64_t) g->mirror_proc_offsets[mpisize];
  for (int64_t k = 0; k < total; ++k) {
    p4est_locidx_t    mi = g->mirror_proc_mirrors[k];
    p8est_quadrant_t *mq = p8est_quadrant_array_index(&g->mirrors, mi);
    out[k] = (int32_t) mq->p.piggy3.local_num;
  }
}
