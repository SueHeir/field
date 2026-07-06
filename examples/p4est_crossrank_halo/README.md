# p4est_crossrank_halo

Validates FIELD's p4est-backed `ForestMesh` cross-rank ghost cells and the
generic `HaloPlan` exchange path. The example runs under two MPI ranks, refines
an AMR patch that crosses the partition, poisons cross-rank ghost cells, and
then checks that `halo_exchange_forward` overwrites every received ghost with
the analytic value implied by that ghost cell's geometry.

```bash
python3 examples/p4est_crossrank_halo/sweep.py
```

![p4est cross-rank halo validation](plots/crossrank_halo.png)

The plot shows the non-vacuous topology checks (received ghosts, changed ghosts,
and cross-rank AMR seam faces) plus the measured ghost-value error against the
configured tolerance. PASS requires all received ghosts to change, at least one
cross-rank AMR face, and `max_abs_err <= 1e-12`.
