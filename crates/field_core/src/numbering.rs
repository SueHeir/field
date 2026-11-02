//! Global row numbering and a distributed CSR matrix over any [`FvMesh`] — the
//! two things an assembled implicit solver (or a PETSc/hypre binding) needs
//! from the mesh.
//!
//! # Numbering
//!
//! [`GlobalNumbering`] gives every owned cell a global row in `0..n_global`,
//! contiguous per rank in rank order (rank `r` owns `first..first + n_owned`,
//! the layout PETSc's `MatCreateAIJ` and hypre's `IJMatrix` expect). Halo
//! cells learn the global row of the cell they mirror through one forward halo
//! exchange. Boundary ghosts (cells no rank owns) have no row.
//!
//! # Matrix
//!
//! [`DistCsr`] stores this rank's rows only, with columns as *local* cell
//! indices (owned or halo), like PETSc's MPIAIJ diagonal + off-diagonal blocks.
//! [`DistCsr::apply`] halo-exchanges `x` and multiplies, so it is a
//! `grass_linalg`-ready operator. [`DistCsr::global_triplets`] exports the rows
//! in global numbering for an external solver.
//!
//! Nothing here depends on the mesh being structured, and the only
//! communication primitives used are the ones GRASS already has (a slice
//! all-reduce and the halo exchange).

use crate::halo::halo_exchange_forward;
use crate::mesh::FvMesh;
use crate::operator::CellVector;
use grass_mpi::CommBackend;

/// Global row ids for the cells of one rank's mesh.
#[derive(Debug, Clone)]
pub struct GlobalNumbering {
    /// First global row this rank owns.
    pub first: u64,
    /// Number of rows this rank owns.
    pub n_owned: usize,
    /// Total rows over all ranks.
    pub n_global: u64,
    /// Global row per local cell (owned and halo); `None` for cells no rank
    /// owns (physical-boundary ghosts).
    rows: Vec<Option<u64>>,
}

impl GlobalNumbering {
    /// Number the owned cells of `mesh` contiguously per rank and fetch the
    /// halo cells' rows from their owners. Collective over `comm`.
    pub fn new(mesh: &dyn FvMesh, comm: &dyn CommBackend) -> Self {
        let n = mesh.n_cells_total();
        let owned: Vec<usize> = (0..n).filter(|&c| mesh.is_local_cell(c)).collect();
        let size = comm.size() as usize;
        let rank = comm.rank() as usize;
        // Exclusive prefix sum of owned counts via one slice all-reduce.
        let mut counts = vec![0.0; size];
        counts[rank] = owned.len() as f64;
        comm.all_reduce_sum_f64_slice(&mut counts);
        let first: u64 = counts[..rank].iter().map(|&c| c as u64).sum();
        let n_global: u64 = counts.iter().map(|&c| c as u64).sum();

        // f64 carries integers exactly up to 2^53, far beyond any mesh.
        let mut ids = CellVector::from_mesh(mesh);
        ids.fill(f64::NAN);
        for (k, &c) in owned.iter().enumerate() {
            ids.set(c, (first + k as u64) as f64);
        }
        halo_exchange_forward(mesh.halo_plan(), &mut ids, comm);
        let rows = (0..n)
            .map(|c| {
                let v = ids.get(c);
                (!v.is_nan()).then_some(v as u64)
            })
            .collect();
        Self {
            first,
            n_owned: owned.len(),
            n_global,
            rows,
        }
    }

    /// Global row of local cell `c`, if any rank owns it.
    pub fn row(&self, c: usize) -> Option<u64> {
        self.rows[c]
    }
}

/// This rank's rows of a distributed sparse matrix over mesh cells.
#[derive(Debug, Clone, Default)]
pub struct DistCsr {
    /// Local (owned) cell of each row.
    pub row_cells: Vec<usize>,
    /// CSR row pointer, `row_cells.len() + 1` long.
    pub row_ptr: Vec<usize>,
    /// Column of each entry, as a local cell index (owned or halo).
    pub col_cells: Vec<usize>,
    /// Value of each entry.
    pub values: Vec<f64>,
}

impl DistCsr {
    /// Build from per-row `(local column cell, value)` lists, merging repeated
    /// columns. `rows[k]` belongs to `row_cells[k]`.
    pub fn from_rows(row_cells: Vec<usize>, rows: Vec<Vec<(usize, f64)>>) -> Self {
        assert_eq!(row_cells.len(), rows.len());
        let mut out = DistCsr {
            row_cells,
            row_ptr: vec![0],
            ..Default::default()
        };
        for mut row in rows {
            row.sort_by_key(|e| e.0);
            for (c, v) in row {
                if out.col_cells.len() > *out.row_ptr.last().expect("non-empty")
                    && *out.col_cells.last().expect("non-empty") == c
                {
                    *out.values.last_mut().expect("non-empty") += v;
                } else {
                    out.col_cells.push(c);
                    out.values.push(v);
                }
            }
            out.row_ptr.push(out.col_cells.len());
        }
        out
    }

    /// `y = A x` over owned cells. `x` is halo-exchanged first (it must be a
    /// mesh-sized [`CellVector`]); entries of `y` outside the rows are zeroed.
    pub fn apply(
        &self,
        mesh: &dyn FvMesh,
        comm: &dyn CommBackend,
        x: &mut CellVector,
        y: &mut [f64],
    ) {
        halo_exchange_forward(mesh.halo_plan(), x, comm);
        let xs = x.as_slice();
        y.fill(0.0);
        for (k, &r) in self.row_cells.iter().enumerate() {
            let mut s = 0.0;
            for e in self.row_ptr[k]..self.row_ptr[k + 1] {
                s += self.values[e] * xs[self.col_cells[e]];
            }
            y[r] = s;
        }
    }

    /// The diagonal entry of each row, by row cell, for a Jacobi preconditioner.
    pub fn diagonal(&self) -> Vec<(usize, f64)> {
        self.row_cells
            .iter()
            .enumerate()
            .map(|(k, &r)| {
                let d = (self.row_ptr[k]..self.row_ptr[k + 1])
                    .find(|&e| self.col_cells[e] == r)
                    .map_or(0.0, |e| self.values[e]);
                (r, d)
            })
            .collect()
    }

    /// This rank's entries as `(global row, global column, value)`, the input
    /// format of an external distributed solver. Panics if a column refers to
    /// a cell no rank owns (a boundary ghost must be folded into the row
    /// before assembly).
    pub fn global_triplets(&self, numbering: &GlobalNumbering) -> Vec<(u64, u64, f64)> {
        let mut out = Vec::with_capacity(self.values.len());
        for (k, &r) in self.row_cells.iter().enumerate() {
            let gr = numbering.row(r).expect("row cell is owned");
            for e in self.row_ptr[k]..self.row_ptr[k + 1] {
                let gc = numbering
                    .row(self.col_cells[e])
                    .expect("column refers to an unowned (boundary ghost) cell");
                out.push((gr, gc, self.values[e]));
            }
        }
        out
    }
}
