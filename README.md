# AD utilities for Futhark  [![CI](https://github.com/diku-dk/autodiff/workflows/CI/badge.svg)](https://github.com/diku-dk/autodiff/actions) [![Documentation](https://futhark-lang.org/pkgs/github.com/diku-dk/autodiff/status.svg)](https://futhark-lang.org/pkgs/github.com/diku-dk/autodiff/latest/)

This is a collection of small utilities for working with AD in
Futhark.

[See examples here.](examples/)

## Installation

```
$ futhark pkg add github.com/diku-dk/autodiff
$ futhark pkg sync
```
## Usage example

```Futhark
> import "lib/github.com/diku-dk/autodiff/onehot"
> import "lib/github.com/diku-dk/autodiff/autodiff"
> grad64 f64.tanh 2.0
7.065082485316443e-2
> grad64 (grad64 f64.tanh) 2.0
-0.13621868742711296
> onehots onehot.(arr (pair f64 f32)) : [][3](f64,f32)
[[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
 [(0.0, 1.0), (0.0, 0.0), (0.0, 0.0)],
 [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)],
 [(0.0, 0.0), (0.0, 1.0), (0.0, 0.0)],
 [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
 [(0.0, 0.0), (0.0, 0.0), (0.0, 1.0)]]
> grad onehot.(arr f64) (map f64.sin) [0, 1, 2]
[[1.0, 0.0, -0.0],
 [0.0, 0.5403023058681398, -0.0],
 [0.0, 0.0, -0.4161468365471424]]
```
