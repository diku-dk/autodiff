-- | ignore

module CSR = import "pattern_csr"
import "sparse_jacobian"

-- One-hot vector
def onehot_f64 [n] (i: i64) : [n]f64 =
  replicate n 0.0f64 with [i] = 1.0f64

-- Dense via JVP (forward mode) with n calls
def jac_dense_jvp [m] [n] (f: [n]f64 -> [m]f64) (x: [n]f64) : [m][n]f64 =
  let cols: [n][m]f64 = map (\j -> jvp f x (onehot_f64 j)) (iota n)
  in transpose cols

-- Dense via VJP (reverse mode) with m calls
def jac_dense_vjp [m] [n] (f: [n]f64 -> [m]f64) (x: [n]f64) : [m][n]f64 =
  map (\i -> vjp f x (onehot_f64 i)) (iota m)

def approx_eq_mat [m] [n] (a: [m][n]f64) (b: [m][n]f64) (eps: f64) : bool =
  let row_ok (ra: [n]f64) (rb: [n]f64): bool = and (map2 (\x y -> f64.abs (x - y) <= eps) ra rb)
  in and (map2 row_ok a b)

def mask_with_pattern [m] [n] (pat: [m][n]bool) (j: [m][n]f64) : [m][n]f64 =
  map2 (\prow jrow ->
          map2 (\p x -> if p then x else 0.0f64) prow jrow)
       pat
       j

def csr_to_dense [m] [n]
                 (row_offs: [m + 1]i64)
                 (row_idx: []i64)
                 (vals: []f64) : [m][n]f64 =
  map (\i ->
         let s = row_offs[i]
         let e = row_offs[i + 1]
         let cols = row_idx[s:e]
         let vs = vals[s:e]
         let row0: [n]f64 = replicate n 0.0f64
         in scatter row0 cols vs)
      (iota m)

-- Example 1: JVP should be preferred.
-- Column coloring needs 2 colors, row coloring needs 4 colors.
def f_jvp_choice (x: [4]f64) : [4]f64 =
  let y0 = x[0] + 2.0f64 * x[1]
  let y1 = x[0] * x[2]
  let y2 = x[0] - x[3] * x[3]
  let y3 = 5.0f64 * x[0]
  in [y0, y1, y2, y3]

def pat_jvp_choice : [4][4]bool =
  [ [true, true, false, false]
  , [true, false, true, false]
  , [true, false, false, true]
  , [true, false, false, false]
  ]

module auto = mk_auto f64

-- JVP-choice dense output: auto selects JVP and matches masked dense Jacobian.
-- ==
-- entry: test_sparse_auto_jvp_choice_dense_with_info
-- input  { [2.0f64, -1.0f64, 3.0f64, 4.0f64] }
-- output { true }
entry test_sparse_auto_jvp_choice_dense_with_info (x: [4]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_jvp_choice (jac_dense_jvp f_jvp_choice x)
  let (ja, use_jvp, num_col_colors, num_row_colors) =
    auto.jac_dense_with_info f_jvp_choice pat_jvp_choice x
  in approx_eq_mat ja jd eps
     && use_jvp
     && num_col_colors == 2i64
     && num_row_colors == 4i64

-- JVP-choice CSR output: auto CSR output reconstructs to masked dense Jacobian.
-- ==
-- entry: test_sparse_auto_jvp_choice_csr_with_info
-- input  { [2.0f64, -1.0f64, 3.0f64, 4.0f64] }
-- output { true }
entry test_sparse_auto_jvp_choice_csr_with_info (x: [4]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_jvp_choice (jac_dense_jvp f_jvp_choice x)
  let ((row_offs, row_idx, vals), use_jvp, num_col_colors, num_row_colors) =
    auto.jac_csr_with_info f_jvp_choice pat_jvp_choice x
  let ja = csr_to_dense row_offs row_idx vals
  in approx_eq_mat ja jd eps
     && use_jvp
     && num_col_colors == 2i64
     && num_row_colors == 4i64

-- Example 2: VJP should be preferred.
-- Column coloring needs 4 colors, row coloring needs 2 colors.
def f_vjp_choice (x: [4]f64) : [4]f64 =
  let y0 = x[0] + x[1] + x[2] + x[3]
  let y1 = 3.0f64 * x[0]
  let y2 = x[1] * x[1]
  let y3 = 4.0f64 * x[2]
  in [y0, y1, y2, y3]

def pat_vjp_choice : [4][4]bool =
  [ [true, true, true, true]
  , [true, false, false, false]
  , [false, true, false, false]
  , [false, false, true, false]
  ]

-- VJP-choice dense output: auto selects VJP and matches masked dense Jacobian.
-- ==
-- entry: test_sparse_auto_vjp_choice_dense_with_info
-- input  { [1.0f64, 2.0f64, 3.0f64, 4.0f64] }
-- output { true }
entry test_sparse_auto_vjp_choice_dense_with_info (x: [4]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_vjp_choice (jac_dense_jvp f_vjp_choice x)
  let (ja, use_jvp, num_col_colors, num_row_colors) =
    auto.jac_dense_with_info f_vjp_choice pat_vjp_choice x
  in approx_eq_mat ja jd eps
     && !use_jvp
     && num_col_colors == 4i64
     && num_row_colors == 2i64

-- VJP-choice CSR output: auto CSR output reconstructs to masked dense Jacobian.
-- ==
-- entry: test_sparse_auto_vjp_choice_csr_with_info
-- input  { [1.0f64, 2.0f64, 3.0f64, 4.0f64] }
-- output { true }
entry test_sparse_auto_vjp_choice_csr_with_info (x: [4]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_vjp_choice (jac_dense_jvp f_vjp_choice x)
  let ((row_offs, row_idx, vals), use_jvp, num_col_colors, num_row_colors) =
    auto.jac_csr_with_info f_vjp_choice pat_vjp_choice x
  let ja = csr_to_dense row_offs row_idx vals
  in approx_eq_mat ja jd eps
     && !use_jvp
     && num_col_colors == 4i64
     && num_row_colors == 2i64

-- Example 3: tie case. Both modes need one color, so JVP should be chosen.
def f_tie_choice (x: [5]f64) : [3]f64 =
  let y0 = 2.0f64 * x[0]
  let y1 = x[2] * x[2]
  let y2 = x[4] - 1.0f64
  in [y0, y1, y2]

def pat_tie_choice : [3][5]bool =
  [ [true, false, false, false, false]
  , [false, false, true, false, false]
  , [false, false, false, false, true]
  ]

-- Tie case: auto chooses JVP when column and row color counts are equal.
-- ==
-- entry: test_sparse_auto_tie_prefers_jvp
-- input  { [1.0f64, 7.0f64, 3.0f64, 9.0f64, -2.0f64] }
-- output { true }
entry test_sparse_auto_tie_prefers_jvp (x: [5]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_tie_choice (jac_dense_jvp f_tie_choice x)
  let (ja, use_jvp, num_col_colors, num_row_colors) =
    auto.jac_dense_with_info f_tie_choice pat_tie_choice x
  in approx_eq_mat ja jd eps
     && use_jvp
     && num_col_colors == 1i64
     && num_row_colors == 1i64

-- Example 4: zero sparsity pattern and constant function.
def f_zero (_x: [4]f64) : [2]f64 =
  [10.0f64, -3.0f64]

def pat_zero : [2][4]bool =
  [ [false, false, false, false]
  , [false, false, false, false]
  ]

-- Zero pattern: auto CSR output reconstructs to the all-zero masked Jacobian.
-- ==
-- entry: test_sparse_auto_zero_pattern_csr
-- input  { [8.0f64, -2.0f64, 5.0f64, 11.0f64] }
-- output { true }
entry test_sparse_auto_zero_pattern_csr (x: [4]f64) : bool =
  let eps = 1e-9f64
  let jd = mask_with_pattern pat_zero (jac_dense_jvp f_zero x)
  let (row_offs, row_idx, vals) =
    auto.jac_csr f_zero pat_zero x
  let ja = csr_to_dense row_offs row_idx vals
  in approx_eq_mat ja jd eps

-- Example 5: mixed nonlinear case with empty row and unused column.
-- VJP should be preferred: column coloring needs 3 colors, row coloring needs 2.
def f_from_csr (x: [6]f64) : [5]f64 =
  let y0 = x[0] * x[1] + x[5]
  let y1 = 2.0f64 * x[2]
  let y2 = x[1] + x[3]
  let y3 = x[0] - x[5] * x[5]
  let y4 = 11.0f64
  in [y0, y1, y2, y3, y4]

def pat_from_csr : [5][6]bool =
  [ [true, true, false, false, false, true]
  , [false, false, true, false, false, false]
  , [false, true, false, true, false, false]
  , [true, false, false, false, false, true]
  , [false, false, false, false, false, false]
  ]

-- CSR-input API: auto uses an existing CSR pattern and selects VJP correctly.
-- ==
-- entry: test_sparse_auto_csr_from_csr_with_info
-- input  { [2.0f64, -3.0f64, 4.0f64, 1.5f64, 99.0f64, -2.0f64] }
-- output { true }
entry test_sparse_auto_csr_from_csr_with_info (x: [6]f64) : bool =
  let eps = 1e-9f64
  let jd =
    mask_with_pattern pat_from_csr (jac_dense_jvp f_from_csr x)
  let ((row_offs, row_idx), (col_offs, col_idx)) =
    CSR.csr_bipartite_from_pattern pat_from_csr
  let ((out_row_offs, out_row_idx, vals), use_jvp, num_col_colors, num_row_colors) =
    auto.jac_csr_from_csr_with_info f_from_csr row_offs row_idx col_offs col_idx x
  let ja =
    csr_to_dense out_row_offs out_row_idx vals
  in approx_eq_mat ja jd eps
     && !use_jvp
     && num_col_colors == 3i64
     && num_row_colors == 2i64
