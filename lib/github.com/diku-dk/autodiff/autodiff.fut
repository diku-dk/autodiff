-- | Various utilities for performing AD.
--
-- The naming here is largely inspired by Barak A. Pearlmutter.

import "onehot"

local def singular 'a (x: onehot.gen [1] a) = onehot.onehot x 0

-- | Compute the tangent of a scalar-valued function given a one-hot
-- generator for its unit domain. The name of this function is a
-- contraction of "derivative of argument".
def darg_unit gen f x = jvp f x (singular gen)

-- | Convenience function for computing the tangent of an
-- 'f32'-valued differentiable function.
def darg32 = darg_unit onehot.f32

-- | Convenience function for computing the tangent of an
-- 'f64'-valued differentiable function.
def darg64 = darg_unit onehot.f64

-- | Compute the gradient of a scalar-valued function given a one-hot
-- generator for its unit result.
def grad_unit gen f x = vjp f x (singular gen)

-- | Convenience function for computing the gradient of an
-- 'f32'-valued differentiable function.
def grad32 = grad_unit onehot.f32

-- | Convenience function for computing the gradient of an
-- 'f64'-valued differentiable function.
def grad64 = grad_unit onehot.f64

-- | Compute the gradient of an arbitrary differentiable function
-- given a one-hot generator for its result.
def grad_rev gen f x = map (vjp f x) (onehots gen)

-- | Hessian-vector product.
def hvp32 f x v =
  jvp (grad32 f) x v

def hvp64 f x v =
  jvp (grad64 f) x v

-- | Compute Jacobian using forward-mode AD.
def jacfwd gen f x =
  map (jvp f x) (onehots gen)

-- | Compute Jacobian using reverse-mode AD.
def jacrev [n] 'a 'b (gen: onehot.gen [n] b) (f: a -> b) (x: a) : [n]a =
  map (vjp f x) (onehots gen)
