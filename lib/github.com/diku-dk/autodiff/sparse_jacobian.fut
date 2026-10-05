-- | Efficiently computing sparse Jacobians.
--
-- This file exposes a lot of facilities, but you probably want to use the
-- `mk_auto` module.
--
-- The modules that provide functionality are parameterised over the number
-- representation. Most users will want to instantiate them with either the
-- `f32` or `f64` modules.
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
def seed_for_color [l] 't (zero: t) (one: t) (colors: [l]i64) (c: i64) : *[l]t =
  map (\i -> if i == c then one else zero) colors

-- | CSR representation of a sparse matrix with n rows and nnz nonzero elements.
type csr [n] [nnz] 't = ([n + 1]i64, [nnz]i64, [nnz]t)

-- | Convert a CSR matrix to a dense matrix. Use this to convert the CSR
-- matrices to dense matrices, if you want.
def csr_to_dense [n] [m] [nnz] 't
                 (zero: t)
                 ((row_offs, row_idx, vals): csr [n] [nnz] t) : *[n][m]t =
  tabulate n \i ->
    let s = row_offs[i]
    let e = row_offs[i + 1]
    let cols = row_idx[s:e]
    let vs = vals[s:e]
    let row0 = replicate m zero
    in scatter row0 cols vs

-- | Facilities for row-wise computation of a Jacobian ("reverse-mode", via
-- vector-Jacobian products).
module mk_rows (R: real)
  : {
      type prepared [n] [m] [nnz] =
        ([m + 1]i64, [nnz]i64, [n + 1]i64, [nnz]i64, [m]i64)

      val prepare [m] [n] :
        [m][n]bool -> ?[nnz].prepared [n] [m] [nnz]

      val prepared_jac_csr [m] [n] [nnz] :
        (f: [n]R.t -> [m]R.t)
        -> prepared [n] [m] [nnz]
        -> (x: [n]R.t)
        -> csr [m] [nnz] R.t

      val jac_csr [m] [n] :
        (f: [n]R.t -> [m]R.t)
        -> (pat: [m][n]bool)
        -> (x: [n]R.t)
        -> ?[nnz].([m + 1]i64, [nnz]i64, [nnz]R.t)
    } = {
  type prepared [n] [m] [nnz] =
    ([m + 1]i64, [nnz]i64, [n + 1]i64, [nnz]i64, [m]i64)

  def compressed_to_csr_vals [m] [n] [nnz] [d]
                             (row_offs: [m + 1]i64)
                             (row_idx: [nnz]i64)
                             (row_colors: [m]i64)
                             (ys: [d][n]R.t) =
    let vals0 = replicate nnz (R.f64 0.0)
    let (vals_final, _i) =
      loop (vals, i) = (vals0, 0)
      while i < m do
        let s = row_offs[i]
        let e = row_offs[i + 1]
        let cols = row_idx[s:e]
        let rc = row_colors[i]
        let seg = map (\j -> ys[rc][j]) cols
        let vals' = vals with [s:e] = seg
        in (vals', i + 1)
    in vals_final

  def compressed_ys [m] [n]
                    (f: [n]R.t -> [m]R.t)
                    (row_colors: [m]i64)
                    (x: [n]R.t) : ?[k].*[k][n]R.t =
    tabulate (num_colors_of row_colors) \c ->
      let seed = seed_for_color (R.f64 0) (R.f64 1) row_colors c
      in vjp f x seed

  -- Precompute everything from an already available CSR sparsity pattern.
  def prepare_from_csr [m] [n] [nnz]
                       (row_offs: [m + 1]i64)
                       (row_idx: []i64)
                       (col_offs: [n + 1]i64)
                       (col_idx: [nnz]i64) : ?[nnz].prepared [n] [m] [nnz] =
    let row_colors =
      Col.partial_d2_color_rows row_offs row_idx col_offs col_idx
    in (row_offs, row_idx, col_offs, col_idx, row_colors)

  -- Precompute everything that only depends on the sparsity pattern.
  def prepare [m] [n]
              (pat: [m][n]bool) : ?[nnz].prepared [n] [m] [nnz] =
    let ((row_offs, row_idx), (col_offs, col_idx)) =
      CSR.csr_bipartite_from_pattern pat
    in prepare_from_csr row_offs row_idx col_offs col_idx

  -- Return the compressed representation using prepared structure/coloring.
  def eval_prepared_compressed [m] [n] [nnz]
                               (f: [n]R.t -> [m]R.t)
                               (prepared: prepared [n] [m] [nnz])
                               (x: [n]R.t) =
    let (row_offs, row_idx, col_offs, col_idx, row_colors) = prepared
    let ys = compressed_ys f row_colors x
    in ((row_offs, row_idx), (col_offs, col_idx), row_colors, ys)

  -- Return the sparse Jacobian in CSR format using prepared structure/coloring.
  def prepared_jac_csr [m] [n] [nnz]
                       (f: [n]R.t -> [m]R.t)
                       (prepared: prepared [n] [m] [nnz])
                       (x: [n]R.t) =
    let ((row_offs, row_idx), (_col_offs, _col_idx), row_colors, ys) =
      eval_prepared_compressed f prepared x
    let vals = compressed_to_csr_vals row_offs row_idx row_colors ys
    in (row_offs, row_idx, vals)

  -- Compressed output:
  -- Returns:
  --   ((row_offs,row_idx), (col_offs,col_idx), row_colors, ys)
  --
  -- This is the compressed Jacobian representation:
  def jac_compressed [m] [n]
                     (f: [n]R.t -> [m]R.t)
                     (pat: [m][n]bool)
                     (x: [n]R.t) =
    let prepared = prepare pat
    in eval_prepared_compressed f prepared x

  -- Sparse / CSR output:
  -- Returns the Jacobian in CSR format:
  --   (row_offs, row_idx, vals)
  def jac_csr [m] [n]
              (f: [n]R.t -> [m]R.t)
              (pat: [m][n]bool)
              (x: [n]R.t) =
    let prepared = prepare pat
    in prepared_jac_csr f prepared x

  -- Compressed output from an already available CSR sparsity pattern.
  def jac_compressed_from_csr [m] [n]
                              (f: [n]R.t -> [m]R.t)
                              (row_offs: [m + 1]i64)
                              (row_idx: []i64)
                              (col_offs: [n + 1]i64)
                              (col_idx: []i64)
                              (x: [n]R.t) =
    let prepared =
      prepare_from_csr row_offs row_idx col_offs col_idx
    in eval_prepared_compressed f prepared x

  -- Sparse / CSR output from an already available CSR sparsity pattern.
  def jac_vjp_csr_from_csr [m] [n]
                           (f: [n]R.t -> [m]R.t)
                           (row_offs: [m + 1]i64)
                           (row_idx: []i64)
                           (col_offs: [n + 1]i64)
                           (col_idx: []i64)
                           (x: [n]R.t) =
    let prepared =
      prepare_from_csr row_offs row_idx col_offs col_idx
    in prepared_jac_csr f prepared x
}

-- | Facilities for column-wise computation of sparse Jacobians ("forward-mode",
-- via Jacobian-vector products).
module mk_cols (R: real)
  : {
      type prepared [n] [m] [nnz] =
        ([m + 1]i64, [nnz]i64, [n + 1]i64, [nnz]i64, [n]i64)

      val prepare [m] [n] :
        [m][n]bool -> ?[nnz].prepared [n] [m] [nnz]

      val prepared_jac_csr [m] [n] [nnz] :
        (f: [n]R.t -> [m]R.t)
        -> prepared [n] [m] [nnz]
        -> (x: [n]R.t)
        -> csr [m] [nnz] R.t

      val jac_csr [m] [n] :
        (f: [n]R.t -> [m]R.t)
        -> (pat: [m][n]bool)
        -> (x: [n]R.t) -> ?[nnz].([m + 1]i64, [nnz]i64, [nnz]R.t)
    } = {
  type prepared [n] [m] [nnz] =
    ([m + 1]i64, [nnz]i64, [n + 1]i64, [nnz]i64, [n]i64)

  -- Reconstruction helpers:
  -- Reconstruct only the nonzero Jacobian values in CSR order.
  -- If row_idx[p] = j belongs to row i, then
  --   vals[p] = J[i,j] = ys[colors[j]][i]
  def compressed_to_csr_vals [m] [n] [nnz] [d]
                             (row_offs: [m + 1]i64)
                             (row_idx: [nnz]i64)
                             (colors: [n]i64)
                             (ys: [d][m]R.t) : [nnz]R.t =
    let vals0 = replicate nnz (R.f64 0)
    let (vals_final, _i) =
      loop (vals, i) = (vals0, 0)
      while i < m do
        let s = row_offs[i]
        let e = row_offs[i + 1]
        let cols = row_idx[s:e]
        let seg = map (\j -> ys[colors[j]][i]) cols
        let vals' = vals with [s:e] = seg
        in (vals', i + 1)
    in vals_final

  def compressed_ys [m] [n]
                    (f: [n]R.t -> [m]R.t)
                    (colors: [n]i64)
                    (x: [n]R.t) : ?[k].[k][m]R.t =
    let nc = num_colors_of colors
    in map (\c ->
              let seed = seed_for_color (R.f64 0) (R.f64 1) colors c
              in jvp f x seed)
           (iota nc)

  -- Precompute everything from an already available CSR sparsity pattern.
  def prepare_from_csr [m] [n]
                       (row_offs: [m + 1]i64)
                       (row_idx: []i64)
                       (col_offs: [n + 1]i64)
                       (col_idx: []i64) =
    let colors =
      Col.partial_d2_color_cols row_offs row_idx col_offs col_idx
    in (row_offs, row_idx, col_offs, col_idx, colors)

  -- Precompute everything that only depends on the sparsity pattern.
  def prepare [m] [n]
              (pat: [m][n]bool) =
    let ((row_offs, row_idx), (col_offs, col_idx)) =
      CSR.csr_bipartite_from_pattern pat
    in prepare_from_csr row_offs row_idx col_offs col_idx

  -- Return the compressed representation using prepared structure/coloring.
  def eval_prepared_compressed [m] [n]
                               (f: [n]R.t -> [m]R.t)
                               (prepared: ([m + 1]i64, []i64, [n + 1]i64, []i64, [n]i64))
                               (x: [n]R.t) =
    let (row_offs, row_idx, col_offs, col_idx, colors) = prepared
    let ys = compressed_ys f colors x
    in ((row_offs, row_idx), (col_offs, col_idx), colors, ys)

  -- Return the sparse Jacobian in CSR format using prepared structure/coloring.
  def prepared_jac_csr [m] [n]
                       (f: [n]R.t -> [m]R.t)
                       (prepared: ([m + 1]i64, []i64, [n + 1]i64, []i64, [n]i64))
                       (x: [n]R.t) =
    let ((row_offs, row_idx), (_col_offs, _col_idx), colors, ys) =
      eval_prepared_compressed f prepared x
    let vals = compressed_to_csr_vals row_offs row_idx colors ys
    in (row_offs, row_idx, vals)

  -- Compressed output:
  -- Returns:
  --   ((row_offs,row_idx), (col_offs,col_idx), colors, ys)
  -- This is the compressed Jacobian representation:
  def jac_compressed [m] [n]
                     (f: [n]R.t -> [m]R.t)
                     (pat: [m][n]bool)
                     (x: [n]R.t) =
    let prepared = prepare pat
    in eval_prepared_compressed f prepared x

  -- Sparse / CSR output:
  -- Returns the Jacobian in CSR format:
  --   (row_offs, row_idx, vals)
  def jac_csr [m] [n]
              (f: [n]R.t -> [m]R.t)
              (pat: [m][n]bool)
              (x: [n]R.t) =
    let prepared = prepare pat
    in prepared_jac_csr f prepared x

  -- Compressed output from an already available CSR sparsity pattern.
  def jac_compressed_from_csr [m] [n]
                              (f: [n]R.t -> [m]R.t)
                              (row_offs: [m + 1]i64)
                              (row_idx: []i64)
                              (col_offs: [n + 1]i64)
                              (col_idx: []i64)
                              (x: [n]R.t) =
    let prepared =
      prepare_from_csr row_offs row_idx col_offs col_idx
    in eval_prepared_compressed f prepared x

  -- Sparse / CSR output from an already available CSR sparsity pattern.
  def jac_csr_from_csr [m] [n]
                       (f: [n]R.t -> [m]R.t)
                       (row_offs: [m + 1]i64)
                       (row_idx: []i64)
                       (col_offs: [n + 1]i64)
                       (col_idx: []i64)
                       (x: [n]R.t) =
    let prepared =
      prepare_from_csr row_offs row_idx col_offs col_idx
    in prepared_jac_csr f prepared x
}

-- | Computation of sparse Jacobians by automatically trying to pick an
-- intelligent approach.
module mk_auto (R: real)
  : {
      val jac_csr [m] [n] :
        (f: [n]R.t -> [m]R.t)
        -> (pat: [m][n]bool)
        -> (x: [n]R.t)
        -> ?[nnz].([m + 1]i64, [nnz]i64, [nnz]R.t)

      val jac_csr_from_csr_with_info [m] [n] [nnz] :
        (f: [n]R.t -> [m]R.t)
        -> (row_offs: [m + 1]i64)
        -> (row_idx: [nnz]i64)
        -> (col_offs: [n + 1]i64)
        -> (col_idx: [nnz]i64)
        -> (x: [n]R.t)
        -> (([m + 1]i64, [nnz]i64, [nnz]R.t), bool, i64, i64)

      -- | Sparse / CSR output with metadata.
      val jac_csr_with_info [m] [n] :
        (f: [n]R.t -> [m]R.t)
        -> (pat: [m][n]bool)
        -> (x: [n]R.t)
        -> ?[nnz].(([m + 1]i64, [nnz]i64, [nnz]R.t), bool, i64, i64)
    } = {
  module rows = mk_rows R
  module cols = mk_cols R

  -- Precompute the structure-dependent part from an already available CSR
  -- sparsity pattern.
  def prepare_jac_from_csr [m] [n]
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
  def prepare_jac [m] [n]
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
    in prepare_jac_from_csr row_offs row_idx col_offs col_idx

  -- Return the sparse Jacobian in CSR format using prepared structure/coloring.
  def prepared_jac_csr [m] [n]
                       (f: [n]R.t -> [m]R.t)
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
                       (x: [n]R.t) =
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
       then cols.prepared_jac_csr f jvp_prepared x
       else rows.prepared_jac_csr f vjp_prepared x

  -- Return the sparse Jacobian in CSR format with metadata.
  def prepared_jac_csr_with_info [m] [n]
                                 (f: [n]R.t -> [m]R.t)
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
                                 (x: [n]R.t) =
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
      prepared_jac_csr f prepared x
    in ((row_offs, row_idx, vals), use_jvp, num_col_colors, num_row_colors)

  -- Return a dense Jacobian using prepared structure/coloring.
  def prepared_jac_dense [m] [n]
                         (f: [n]R.t -> [m]R.t)
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
                         (x: [n]R.t) : [m][n]R.t =
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
    in csr_to_dense (R.f64 0)
                    (if use_jvp
                     then cols.prepared_jac_csr f jvp_prepared x
                     else rows.prepared_jac_csr f vjp_prepared x)

  -- Return a dense Jacobian with metadata.
  def prepared_jac_dense_with_info [m] [n]
                                   (f: [n]R.t -> [m]R.t)
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
                                   (x: [n]R.t) : ([m][n]R.t, bool, i64, i64) =
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
      prepared_jac_dense f prepared x
    in (jac, use_jvp, num_col_colors, num_row_colors)

  -- Returns only the chosen mode and coloring info.
  def jac_choice [m] [n]
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
      prepare_jac pat
    in (use_jvp, num_col_colors, num_row_colors)

  -- Sparse / CSR output.
  def jac_csr [m] [n]
              (f: [n]R.t -> [m]R.t)
              (pat: [m][n]bool)
              (x: [n]R.t) =
    let prepared = prepare_jac pat
    in prepared_jac_csr f prepared x

  -- Sparse / CSR output with metadata.
  def jac_csr_with_info [m] [n]
                        (f: [n]R.t -> [m]R.t)
                        (pat: [m][n]bool)
                        (x: [n]R.t) =
    let prepared = prepare_jac pat
    in prepared_jac_csr_with_info f prepared x

  -- Sparse / CSR output from an already available CSR sparsity pattern.
  def jac_csr_from_csr [m] [n]
                       (f: [n]R.t -> [m]R.t)
                       (row_offs: [m + 1]i64)
                       (row_idx: []i64)
                       (col_offs: [n + 1]i64)
                       (col_idx: []i64)
                       (x: [n]R.t) =
    let prepared =
      prepare_jac_from_csr row_offs row_idx col_offs col_idx
    in prepared_jac_csr f prepared x

  -- Sparse / CSR output with metadata from an already available CSR sparsity
  -- pattern.
  def jac_csr_from_csr_with_info [m] [n]
                                 (f: [n]R.t -> [m]R.t)
                                 (row_offs: [m + 1]i64)
                                 (row_idx: []i64)
                                 (col_offs: [n + 1]i64)
                                 (col_idx: []i64)
                                 (x: [n]R.t) =
    let prepared =
      prepare_jac_from_csr row_offs row_idx col_offs col_idx
    in prepared_jac_csr_with_info f prepared x

  -- Dense output with metadata.
  def jac_dense_with_info [m] [n]
                          (f: [n]R.t -> [m]R.t)
                          (pat: [m][n]bool)
                          (x: [n]R.t) : ([m][n]R.t, bool, i64, i64) =
    let prepared = prepare_jac pat
    in prepared_jac_dense_with_info f prepared x
}
