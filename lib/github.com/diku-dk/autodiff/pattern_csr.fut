-- | Compute CSR (Compressed Sparse Row) from a dense boolean sparsity pattern.

local
def count_true [n] (row: [n]bool) : i64 =
  reduce (+) 0 (map (\b -> if b then 1i64 else 0i64) row)

local
def counts_to_offs [m] (counts: [m]i64) : [m + 1]i64 =
  let pref: [m]i64 = scan (+) 0 counts
  in replicate (m + 1) 0i64 with [1:m + 1] = pref

def csr_rows_from_pattern [m] [n] (pat: [m][n]bool) : ([m + 1]i64, []i64) =
  let counts: [m]i64 = map count_true pat
  -- Work O(m*n)
  let offs: [m + 1]i64 = counts_to_offs counts
  let mask: [m * n]bool = flatten pat
  let cols_flat: [m * n]i64 = map (\k -> k % n) (iota (m * n))
  let pairs: [m * n](bool, i64) = map2 (\b c -> (b, c)) mask cols_flat
  -- Space O(m*n)
  let kept: [](bool, i64) = filter (\(b, _) -> b) pairs
  -- Span O(log(m*n))
  let idx: []i64 = map (.1) kept
  in (offs, idx)

def csr_cols_from_pattern [m] [n] (pat: [m][n]bool) : ([n + 1]i64, []i64) =
  csr_rows_from_pattern (transpose pat)

def csr_bipartite_from_pattern [m] [n] (pat: [m][n]bool) : ?[nnz].(([m + 1]i64, [nnz]i64), ([n + 1]i64, [nnz]i64)) =
  let [nnz] rows: ([m + 1]i64, [nnz]i64) = csr_rows_from_pattern pat
  let cols = csr_cols_from_pattern pat
  -- We need size coercion as the type checker cannot figure out these are the
  -- same.
  in (rows, cols :> ([n + 1]i64, [nnz]i64))
