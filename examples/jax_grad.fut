-- The following example is ported from the Jax cookbook at
-- https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html

import "../lib/github.com/diku-dk/autodiff/autodiff"

def dot a b = f64.sum (map2 (*) a b)

def sigmoid x =
  0.5 * (f64.tanh(x/2)+1)

def predict W b inputs =
  sigmoid(dot inputs W + b)

def inputs : [][]f64 =
  [[0.52, 1.12,  0.77],
   [0.88, -1.08, 0.15],
   [0.52, 0.06, -1.30],
   [0.74, -2.49, 1.39]]

def targets =
  [true,true,false,true]

def vecadd = map2 (f64.+)
def vecmul = map2 (f64.*)

def loss W b =
  let preds = map (predict W b) inputs
  let label_probs = (preds `vecmul` map f64.bool targets)
                         `vecadd` (map (1-) preds `vecmul`
                                       map ((1-) <-< f64.bool) targets)
  in -f64.sum(map f64.log label_probs)

-- Not going to import random number generation just for this. These
-- are made with 'futhark dataset'.
def W = [0.12152684143560777f64, 0.5526745035133085f64, 0.5189896463245001f64]
def b = 0.12152684143560777f64

def Wb_grad = grad64 (\(W,b) -> loss W b) (W,b)
def W_grad = Wb_grad.0
def b_grad = Wb_grad.1
