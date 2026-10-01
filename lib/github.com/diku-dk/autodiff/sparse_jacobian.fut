-- | Efficiently computing sparse Jacobians.
--
-- This file exposes a lot of facilities, but you probably want to use the
-- `auto` module.
--
-- ## Acknowledgements
--
-- Based on work by Elias Smedegaard.

local module CSR = import "pattern_csr"
local module Col = import "partial_d2_coloring"

local
def num_colors_of [l] (colors: [l]i64) : i64 =
  if l == 0
  then 0
  else 1 + i64.maximum colors

local
def seed_for_color [l] (colors: [l]i64) (c: i64) : [l]f64 =
  map (\i -> if i == c then 1 else 0) colors

-- Convert a CSR matrix to a dense matrix.
local
def csr_to_dense [m] [n]
                 (row_offs: [m + 1]i64)
                 (row_idx: []i64)
                 (vals: []f64) : [m][n]f64 =
  tabulate m \i ->
    let s = row_offs[i]
    let e = row_offs[i + 1]
    let cols = row_idx[s:e]
    let vs = vals[s:e]
    let row0: [n]f64 = replicate n 0.0f64
    in scatter row0 cols vs

-- | Facilities for row-wise computation of a Jacobian ("reverse-mode", via
-- vector-Jacobian products).
module rows
  : {
      type prepared [n] [m] [nnz] =
        ([m + 1]i64, [nnz]i64, [n + 1]i64, [nnz]i64, [m]i64)

      val eval_prepared_vjp_csr [m] [n] [nnz] :
        (f: [n]f64 -> [m]f64)
        -> prepared [n] [m] [nnz]
        -> (x: [n]f64)
        -> ([m + 1]i64, [nnz]i64, [nnz]f64)

      val eval_prepared_vjp_dense [m] [n] [nnz] :
        (f: [n]f64 -> [m]f64)
        -> (prepared: ([m + 1]i64, [nnz]i64, [n + 1]i64, [nnz]i64, [m]i64))
        -> (x: [n]f64) -> [m][n]f64
    } = {
  type prepared [n] [m] [nnz] =
    ([m + 1]i64, [nnz]i64, [n + 1]i64, [nnz]i64, [m]i64)

  def vjp_compressed_to_csr_vals [m] [n] [nnz] [d]
                                 (row_offs: [m + 1]i64)
                                 (row_idx: [nnz]i64)
                                 (row_colors: [m]i64)
                                 (ys: [d][n]f64) : [nnz]f64 =
    let vals0 = replicate nnz 0.0f64
    let (vals_final, _i) =
      loop (vals, i) = (vals0, 0i64)
      while i < m do
        let s = row_offs[i]
        let e = row_offs[i + 1]
        let cols = row_idx[s:e]
        let rc = row_colors[i]
        let seg = map (\j -> ys[rc][j]) cols
        let vals' = vals with [s:e] = seg
        in (vals', i + 1i64)
    in vals_final

  def compressed_ys_vjp [m] [n]
                        (f: [n]f64 -> [m]f64)
                        (row_colors: [m]i64)
                        (x: [n]f64) : ?[k].[k][n]f64 =
    tabulate (num_colors_of row_colors) \c ->
      let seed = seed_for_color row_colors c
      in vjp f x seed

  -- Precompute everything from an already available CSR sparsity pattern.
  def prepare_vjp_from_csr [m] [n]
                           (row_offs: [m + 1]i64)
                           (row_idx: []i64)
                           (col_offs: [n + 1]i64)
                           (col_idx: []i64) =
    let row_colors =
      Col.partial_d2_color_rows row_offs row_idx col_offs col_idx
    in (row_offs, row_idx, col_offs, col_idx, row_colors)

  -- Precompute everything that only depends on the sparsity pattern.
  def prepare_vjp [m] [n]
                  (pat: [m][n]bool) =
    let ((row_offs, row_idx), (col_offs, col_idx)) =
      CSR.csr_bipartite_from_pattern pat
    in prepare_vjp_from_csr row_offs row_idx col_offs col_idx

  -- Return the compressed representation using prepared structure/coloring.
  def eval_prepared_vjp_compressed [m] [n]
                                   (f: [n]f64 -> [m]f64)
                                   (prepared: ([m + 1]i64, []i64, [n + 1]i64, []i64, [m]i64))
                                   (x: [n]f64) =
    let (row_offs, row_idx, col_offs, col_idx, row_colors) = prepared
    let ys = compressed_ys_vjp f row_colors x
    in ((row_offs, row_idx), (col_offs, col_idx), row_colors, ys)

  -- Return the sparse Jacobian in CSR format using prepared structure/coloring.
  def eval_prepared_vjp_csr [m] [n]
                            (f: [n]f64 -> [m]f64)
                            (prepared: ([m + 1]i64, []i64, [n + 1]i64, []i64, [m]i64))
                            (x: [n]f64) =
    let ((row_offs, row_idx), (_col_offs, _col_idx), row_colors, ys) =
      eval_prepared_vjp_compressed f prepared x
    let vals = vjp_compressed_to_csr_vals row_offs row_idx row_colors ys
    in (row_offs, row_idx, vals)

  -- Return a dense Jacobian using prepared structure/coloring.
  def eval_prepared_vjp_dense [m] [n]
                              (f: [n]f64 -> [m]f64)
                              (prepared: ([m + 1]i64, []i64, [n + 1]i64, []i64, [m]i64))
                              (x: [n]f64) : [m][n]f64 =
    let (row_offs, row_idx, vals) =
      eval_prepared_vjp_csr f prepared x
    in csr_to_dense row_offs row_idx vals

  -- Compressed output:
  -- Returns:
  --   ((row_offs,row_idx), (col_offs,col_idx), row_colors, ys)
  --
  -- This is the compressed Jacobian representation:
  def jac_vjp_compressed [m] [n]
                         (f: [n]f64 -> [m]f64)
                         (pat: [m][n]bool)
                         (x: [n]f64) =
    let prepared = prepare_vjp pat
    in eval_prepared_vjp_compressed f prepared x

  -- Sparse / CSR output:
  -- Returns the Jacobian in CSR format:
  --   (row_offs, row_idx, vals)
  def jac_vjp_csr [m] [n]
                  (f: [n]f64 -> [m]f64)
                  (pat: [m][n]bool)
                  (x: [n]f64) =
    let prepared = prepare_vjp pat
    in eval_prepared_vjp_csr f prepared x

  -- Dense output:
  def jac_vjp_dense [m] [n]
                    (f: [n]f64 -> [m]f64)
                    (pat: [m][n]bool)
                    (x: [n]f64) : [m][n]f64 =
    let prepared = prepare_vjp pat
    in eval_prepared_vjp_dense f prepared x

  -- Compressed output from an already available CSR sparsity pattern.
  def jac_vjp_compressed_from_csr [m] [n]
                                  (f: [n]f64 -> [m]f64)
                                  (row_offs: [m + 1]i64)
                                  (row_idx: []i64)
                                  (col_offs: [n + 1]i64)
                                  (col_idx: []i64)
                                  (x: [n]f64) =
    let prepared =
      prepare_vjp_from_csr row_offs row_idx col_offs col_idx
    in eval_prepared_vjp_compressed f prepared x

  -- Sparse / CSR output from an already available CSR sparsity pattern.
  def jac_vjp_csr_from_csr [m] [n]
                           (f: [n]f64 -> [m]f64)
                           (row_offs: [m + 1]i64)
                           (row_idx: []i64)
                           (col_offs: [n + 1]i64)
                           (col_idx: []i64)
                           (x: [n]f64) =
    let prepared =
      prepare_vjp_from_csr row_offs row_idx col_offs col_idx
    in eval_prepared_vjp_csr f prepared x

  -- Like jac_vjp_compressed, but assumes row colors are already computed.
  def jac_vjp_compressed_with_row_colors [m] [n]
                                         (f: [n]f64 -> [m]f64)
                                         (pat: [m][n]bool)
                                         (row_colors: [m]i64)
                                         (x: [n]f64) =
    let ((row_offs, row_idx), (col_offs, col_idx)) =
      CSR.csr_bipartite_from_pattern pat
    let ys = compressed_ys_vjp f row_colors x
    in ((row_offs, row_idx), (col_offs, col_idx), row_colors, ys)

  -- Like jac_vjp_csr, but assumes row colors are already computed.
  def jac_vjp_csr_with_row_colors [m] [n]
                                  (f: [n]f64 -> [m]f64)
                                  (pat: [m][n]bool)
                                  (row_colors: [m]i64)
                                  (x: [n]f64) =
    let ((row_offs, row_idx), (_col_offs, _col_idx), _row_colors, ys) =
      jac_vjp_compressed_with_row_colors f pat row_colors x
    let vals = vjp_compressed_to_csr_vals row_offs row_idx row_colors ys
    in (row_offs, row_idx, vals)

  -- Like jac_vjp_dense, but assumes row colors are already computed
  def jac_vjp_dense_with_row_colors [m] [n]
                                    (f: [n]f64 -> [m]f64)
                                    (pat: [m][n]bool)
                                    (row_colors: [m]i64)
                                    (x: [n]f64) : [m][n]f64 =
    let (row_offs, row_idx, vals) =
      jac_vjp_csr_with_row_colors f pat row_colors x
    in csr_to_dense row_offs row_idx vals
}

-- | Facilities for column-wise computation of sparse Jacobians ("forward-mode",
-- via Jacobian-vector products).
module cols
  : {
      type prepared [n] [m] [nnz] =
        ([m + 1]i64, [nnz]i64, [n + 1]i64, [nnz]i64, [n]i64)

      val eval_prepared_jvp_csr [m] [n] [nnz] :
        (f: [n]f64 -> [m]f64)
        -> prepared [n] [m] [nnz]
        -> (x: [n]f64)
        -> ([m + 1]i64, [nnz]i64, [nnz]f64)

      val eval_prepared_jvp_dense [m] [n] [nnz] [b] :
        (f: [n]f64 -> [m]f64)
        -> (prepared: ([m + 1]i64, [nnz]i64, [n + 1]i64, [b]i64, [n]i64))
        -> (x: [n]f64) -> [m][n]f64
    } = {
  type prepared [n] [m] [nnz] =
    ([m + 1]i64, [nnz]i64, [n + 1]i64, [nnz]i64, [n]i64)

  -- Reconstruction helpers:
  -- Reconstruct only the nonzero Jacobian values in CSR order.
  -- If row_idx[p] = j belongs to row i, then
  --   vals[p] = J[i,j] = ys[colors[j]][i]
  def jvp_compressed_to_csr_vals [m] [n] [nnz] [d]
                                 (row_offs: [m + 1]i64)
                                 (row_idx: [nnz]i64)
                                 (colors: [n]i64)
                                 (ys: [d][m]f64) : [nnz]f64 =
    let vals0 = replicate nnz 0.0f64
    let (vals_final, _i) =
      loop (vals, i) = (vals0, 0i64)
      while i < m do
        let s = row_offs[i]
        let e = row_offs[i + 1]
        let cols = row_idx[s:e]
        let seg = map (\j -> ys[colors[j]][i]) cols
        let vals' = vals with [s:e] = seg
        in (vals', i + 1i64)
    in vals_final

  def compressed_ys_jvp [m] [n]
                        (f: [n]f64 -> [m]f64)
                        (colors: [n]i64)
                        (x: [n]f64) : ?[k].[k][m]f64 =
    let nc = num_colors_of colors
    in map (\c ->
              let seed = seed_for_color colors c
              in jvp f x seed)
           (iota nc)

  -- Precompute everything from an already available CSR sparsity pattern.
  def prepare_jvp_from_csr [m] [n]
                           (row_offs: [m + 1]i64)
                           (row_idx: []i64)
                           (col_offs: [n + 1]i64)
                           (col_idx: []i64) =
    let colors =
      Col.partial_d2_color_cols row_offs row_idx col_offs col_idx
    in (row_offs, row_idx, col_offs, col_idx, colors)

  -- Precompute everything that only depends on the sparsity pattern.
  def prepare_jvp [m] [n]
                  (pat: [m][n]bool) =
    let ((row_offs, row_idx), (col_offs, col_idx)) =
      CSR.csr_bipartite_from_pattern pat
    in prepare_jvp_from_csr row_offs row_idx col_offs col_idx

  -- Return the compressed representation using prepared structure/coloring.
  def eval_prepared_jvp_compressed [m] [n]
                                   (f: [n]f64 -> [m]f64)
                                   (prepared: ([m + 1]i64, []i64, [n + 1]i64, []i64, [n]i64))
                                   (x: [n]f64) =
    let (row_offs, row_idx, col_offs, col_idx, colors) = prepared
    let ys = compressed_ys_jvp f colors x
    in ((row_offs, row_idx), (col_offs, col_idx), colors, ys)

  -- Return the sparse Jacobian in CSR format using prepared structure/coloring.
  def eval_prepared_jvp_csr [m] [n]
                            (f: [n]f64 -> [m]f64)
                            (prepared: ([m + 1]i64, []i64, [n + 1]i64, []i64, [n]i64))
                            (x: [n]f64) =
    let ((row_offs, row_idx), (_col_offs, _col_idx), colors, ys) =
      eval_prepared_jvp_compressed f prepared x
    let vals = jvp_compressed_to_csr_vals row_offs row_idx colors ys
    in (row_offs, row_idx, vals)

  -- Return a dense Jacobian using prepared structure/coloring.
  def eval_prepared_jvp_dense [m] [n]
                              (f: [n]f64 -> [m]f64)
                              (prepared: ([m + 1]i64, []i64, [n + 1]i64, []i64, [n]i64))
                              (x: [n]f64) : [m][n]f64 =
    let (row_offs, row_idx, vals) =
      eval_prepared_jvp_csr f prepared x
    in csr_to_dense row_offs row_idx vals

  -- Compressed output:
  -- Returns:
  --   ((row_offs,row_idx), (col_offs,col_idx), colors, ys)
  -- This is the compressed Jacobian representation:
  def jac_jvp_compressed [m] [n]
                         (f: [n]f64 -> [m]f64)
                         (pat: [m][n]bool)
                         (x: [n]f64) =
    let prepared = prepare_jvp pat
    in eval_prepared_jvp_compressed f prepared x

  -- Sparse / CSR output:
  -- Returns the Jacobian in CSR format:
  --   (row_offs, row_idx, vals)
  def jac_jvp_csr [m] [n]
                  (f: [n]f64 -> [m]f64)
                  (pat: [m][n]bool)
                  (x: [n]f64) =
    let prepared = prepare_jvp pat
    in eval_prepared_jvp_csr f prepared x

  -- Dense output:
  def jac_jvp_dense [m] [n]
                    (f: [n]f64 -> [m]f64)
                    (pat: [m][n]bool)
                    (x: [n]f64) : [m][n]f64 =
    let prepared = prepare_jvp pat
    in eval_prepared_jvp_dense f prepared x

  -- Compressed output from an already available CSR sparsity pattern.
  def jac_jvp_compressed_from_csr [m] [n]
                                  (f: [n]f64 -> [m]f64)
                                  (row_offs: [m + 1]i64)
                                  (row_idx: []i64)
                                  (col_offs: [n + 1]i64)
                                  (col_idx: []i64)
                                  (x: [n]f64) =
    let prepared =
      prepare_jvp_from_csr row_offs row_idx col_offs col_idx
    in eval_prepared_jvp_compressed f prepared x

  -- Sparse / CSR output from an already available CSR sparsity pattern.
  def jac_jvp_csr_from_csr [m] [n]
                           (f: [n]f64 -> [m]f64)
                           (row_offs: [m + 1]i64)
                           (row_idx: []i64)
                           (col_offs: [n + 1]i64)
                           (col_idx: []i64)
                           (x: [n]f64) =
    let prepared =
      prepare_jvp_from_csr row_offs row_idx col_offs col_idx
    in eval_prepared_jvp_csr f prepared x

  -- Like jac_jvp_compressed, but assumes colors are already computed.
  def jac_jvp_compressed_with_colors [m] [n]
                                     (f: [n]f64 -> [m]f64)
                                     (pat: [m][n]bool)
                                     (colors: [n]i64)
                                     (x: [n]f64) =
    let ((row_offs, row_idx), (col_offs, col_idx)) =
      CSR.csr_bipartite_from_pattern pat
    let ys = compressed_ys_jvp f colors x
    in ((row_offs, row_idx), (col_offs, col_idx), colors, ys)

  -- Like jac_jvp_csr, but assumes colors are already computed.
  def jac_jvp_csr_with_colors [m] [n]
                              (f: [n]f64 -> [m]f64)
                              (pat: [m][n]bool)
                              (colors: [n]i64)
                              (x: [n]f64) =
    let ((row_offs, row_idx), (_col_offs, _col_idx), _colors, ys) =
      jac_jvp_compressed_with_colors f pat colors x
    let vals = jvp_compressed_to_csr_vals row_offs row_idx colors ys
    in (row_offs, row_idx, vals)

  -- Like jac_jvp_dense, but assumes colors are already computed
  def jac_jvp_dense_with_colors [m] [n]
                                (f: [n]f64 -> [m]f64)
                                (pat: [m][n]bool)
                                (colors: [n]i64)
                                (x: [n]f64) : [m][n]f64 =
    let (row_offs, row_idx, vals) =
      jac_jvp_csr_with_colors f pat colors x
    in csr_to_dense row_offs row_idx vals
}

-- | Computation of sparse Jacobians by automatically trying to pick an
-- intelligent approach.
module auto
  : {
      val jac_auto_csr [m] [n] :
        (f: [n]f64 -> [m]f64)
        -> (pat: [m][n]bool)
        -> (x: [n]f64)
        -> ?[nnz].([m + 1]i64, [nnz]i64, [nnz]f64)

      val jac_auto_dense_with_info [m] [n] :
        (f: [n]f64 -> [m]f64)
        -> (pat: [m][n]bool)
        -> (x: [n]f64)
        -> ([m][n]f64, bool, i64, i64)

      val jac_auto_csr_from_csr_with_info [m] [n] [nnz] :
        (f: [n]f64 -> [m]f64)
        -> (row_offs: [m + 1]i64)
        -> (row_idx: [nnz]i64)
        -> (col_offs: [n + 1]i64)
        -> (col_idx: [nnz]i64)
        -> (x: [n]f64)
        -> (([m + 1]i64, [nnz]i64, [nnz]f64), bool, i64, i64)

      -- | Sparse / CSR output with metadata.
      val jac_auto_csr_with_info [m] [n] :
        (f: [n]f64 -> [m]f64)
        -> (pat: [m][n]bool)
        -> (x: [n]f64)
        -> ?[nnz].(([m + 1]i64, [nnz]i64, [nnz]f64), bool, i64, i64)
    } = {
  -- Precompute the structure-dependent part from an already available CSR
  -- sparsity pattern.
  def prepare_jac_auto_from_csr [m] [n]
                                (row_offs: [m + 1]i64)
                                (row_idx: []i64)
                                (col_offs: [n + 1]i64)
                                (col_idx: []i64) : ( [m + 1]i64
                                                   , []i64
                                                   , [n + 1]i64
                                                   , []i64
                                                   , [n]i64
                                                   , [m]i64
                                                   , i64
                                                   , i64
                                                   , bool
                                                   ) =
    let col_colors =
      Col.partial_d2_color_cols row_offs row_idx col_offs col_idx
    let row_colors =
      Col.partial_d2_color_rows row_offs row_idx col_offs col_idx
    let num_col_colors = num_colors_of col_colors
    let num_row_colors = num_colors_of row_colors
    let use_jvp = num_col_colors <= num_row_colors
    in ( row_offs
       , row_idx
       , col_offs
       , col_idx
       , col_colors
       , row_colors
       , num_col_colors
       , num_row_colors
       , use_jvp
       )

  -- Precompute the structure-dependent part of the automatic sparse Jacobian
  -- pipeline. This computes CSR structure, both colorings, the number of
  -- colors, and whether JVP or VJP should be used.
  --
  -- use_jvp = true  => choose JVP
  -- use_jvp = false => choose VJP
  def prepare_jac_auto [m] [n]
                       (pat: [m][n]bool) : ( [m + 1]i64
                                           , []i64
                                           , [n + 1]i64
                                           , []i64
                                           , [n]i64
                                           , [m]i64
                                           , i64
                                           , i64
                                           , bool
                                           ) =
    let ((row_offs, row_idx), (col_offs, col_idx)) =
      CSR.csr_bipartite_from_pattern pat
    in prepare_jac_auto_from_csr row_offs row_idx col_offs col_idx

  -- Return the sparse Jacobian in CSR format using prepared structure/coloring.
  def eval_prepared_auto_csr [m] [n]
                             (f: [n]f64 -> [m]f64)
                             (prepared: ( [m + 1]i64
                                        , []i64
                                        , [n + 1]i64
                                        , []i64
                                        , [n]i64
                                        , [m]i64
                                        , i64
                                        , i64
                                        , bool
                                        )
                             )
                             (x: [n]f64) =
    let ( row_offs
        , row_idx
        , col_offs
        , col_idx
        , col_colors
        , row_colors
        , _num_col_colors
        , _num_row_colors
        , use_jvp
        ) =
      prepared
    let jvp_prepared = (row_offs, row_idx, col_offs, col_idx, col_colors)
    let vjp_prepared = (row_offs, row_idx, col_offs, col_idx, row_colors)
    in if use_jvp
       then cols.eval_prepared_jvp_csr f jvp_prepared x
       else rows.eval_prepared_vjp_csr f vjp_prepared x

  -- Return the sparse Jacobian in CSR format with metadata.
  def eval_prepared_auto_csr_with_info [m] [n]
                                       (f: [n]f64 -> [m]f64)
                                       (prepared: ( [m + 1]i64
                                                  , []i64
                                                  , [n + 1]i64
                                                  , []i64
                                                  , [n]i64
                                                  , [m]i64
                                                  , i64
                                                  , i64
                                                  , bool
                                                  )
                                       )
                                       (x: [n]f64) =
    let ( _row_offs
        , _row_idx
        , _col_offs
        , _col_idx
        , _col_colors
        , _row_colors
        , num_col_colors
        , num_row_colors
        , use_jvp
        ) =
      prepared
    let (row_offs, row_idx, vals) =
      eval_prepared_auto_csr f prepared x
    in ((row_offs, row_idx, vals), use_jvp, num_col_colors, num_row_colors)

  -- Return a dense Jacobian using prepared structure/coloring.
  def eval_prepared_auto_dense [m] [n]
                               (f: [n]f64 -> [m]f64)
                               (prepared: ( [m + 1]i64
                                          , []i64
                                          , [n + 1]i64
                                          , []i64
                                          , [n]i64
                                          , [m]i64
                                          , i64
                                          , i64
                                          , bool
                                          )
                               )
                               (x: [n]f64) : [m][n]f64 =
    let ( row_offs
        , row_idx
        , col_offs
        , col_idx
        , col_colors
        , row_colors
        , _num_col_colors
        , _num_row_colors
        , use_jvp
        ) =
      prepared
    let jvp_prepared = (row_offs, row_idx, col_offs, col_idx, col_colors)
    let vjp_prepared = (row_offs, row_idx, col_offs, col_idx, row_colors)
    in if use_jvp
       then cols.eval_prepared_jvp_dense f jvp_prepared x
       else rows.eval_prepared_vjp_dense f vjp_prepared x

  -- Return a dense Jacobian with metadata.
  def eval_prepared_auto_dense_with_info [m] [n]
                                         (f: [n]f64 -> [m]f64)
                                         (prepared: ( [m + 1]i64
                                                    , []i64
                                                    , [n + 1]i64
                                                    , []i64
                                                    , [n]i64
                                                    , [m]i64
                                                    , i64
                                                    , i64
                                                    , bool
                                                    )
                                         )
                                         (x: [n]f64) : ([m][n]f64, bool, i64, i64) =
    let ( _row_offs
        , _row_idx
        , _col_offs
        , _col_idx
        , _col_colors
        , _row_colors
        , num_col_colors
        , num_row_colors
        , use_jvp
        ) =
      prepared
    let jac =
      eval_prepared_auto_dense f prepared x
    in (jac, use_jvp, num_col_colors, num_row_colors)

  -- Returns only the chosen mode and coloring info.
  def jac_auto_choice [m] [n]
                      (pat: [m][n]bool) : (bool, i64, i64) =
    let ( _row_offs
        , _row_idx
        , _col_offs
        , _col_idx
        , _col_colors
        , _row_colors
        , num_col_colors
        , num_row_colors
        , use_jvp
        ) =
      prepare_jac_auto pat
    in (use_jvp, num_col_colors, num_row_colors)

  -- Sparse / CSR output.
  def jac_auto_csr [m] [n]
                   (f: [n]f64 -> [m]f64)
                   (pat: [m][n]bool)
                   (x: [n]f64) =
    let prepared = prepare_jac_auto pat
    in eval_prepared_auto_csr f prepared x

  -- Sparse / CSR output with metadata.
  def jac_auto_csr_with_info [m] [n]
                             (f: [n]f64 -> [m]f64)
                             (pat: [m][n]bool)
                             (x: [n]f64) =
    let prepared = prepare_jac_auto pat
    in eval_prepared_auto_csr_with_info f prepared x

  -- Sparse / CSR output from an already available CSR sparsity pattern.
  def jac_auto_csr_from_csr [m] [n]
                            (f: [n]f64 -> [m]f64)
                            (row_offs: [m + 1]i64)
                            (row_idx: []i64)
                            (col_offs: [n + 1]i64)
                            (col_idx: []i64)
                            (x: [n]f64) =
    let prepared =
      prepare_jac_auto_from_csr row_offs row_idx col_offs col_idx
    in eval_prepared_auto_csr f prepared x

  -- Sparse / CSR output with metadata from an already available CSR sparsity
  -- pattern.
  def jac_auto_csr_from_csr_with_info [m] [n]
                                      (f: [n]f64 -> [m]f64)
                                      (row_offs: [m + 1]i64)
                                      (row_idx: []i64)
                                      (col_offs: [n + 1]i64)
                                      (col_idx: []i64)
                                      (x: [n]f64) =
    let prepared =
      prepare_jac_auto_from_csr row_offs row_idx col_offs col_idx
    in eval_prepared_auto_csr_with_info f prepared x

  -- Dense output.
  def jac_auto_dense [m] [n]
                     (f: [n]f64 -> [m]f64)
                     (pat: [m][n]bool)
                     (x: [n]f64) : [m][n]f64 =
    let prepared = prepare_jac_auto pat
    in eval_prepared_auto_dense f prepared x

  -- Dense output with metadata.
  def jac_auto_dense_with_info [m] [n]
                               (f: [n]f64 -> [m]f64)
                               (pat: [m][n]bool)
                               (x: [n]f64) : ([m][n]f64, bool, i64, i64) =
    let prepared = prepare_jac_auto pat
    in eval_prepared_auto_dense_with_info f prepared x
}
