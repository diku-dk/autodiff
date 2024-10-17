# AD utilities for Futhark  [![CI](https://github.com/diku-dk/autodiff/workflows/CI/badge.svg)](https://github.com/diku-dk/autodiff/actions) [![Documentation](https://futhark-lang.org/pkgs/github.com/diku-dk/autodiff/status.svg)](https://futhark-lang.org/pkgs/github.com/diku-dk/autodiff/latest/)

This is a collection of small utilities for working with AD in
Futhark.

## Installation

```
$ futhark pkg add github.com/diku-dk/autodiff
$ futhark pkg sync
```
## Usage example

```Futhark
> import "lib/github.com/diku-dk/autodiff/onehot"
> onehots (onehot.(arr (pair f64 f32))) : [][3](f64,f32)
[[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
 [(0.0, 1.0), (0.0, 0.0), (0.0, 0.0)],
 [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)],
 [(0.0, 0.0), (0.0, 1.0), (0.0, 0.0)],
 [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
 [(0.0, 0.0), (0.0, 0.0), (0.0, 1.0)]]

```
