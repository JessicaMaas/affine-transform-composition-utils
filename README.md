# affine-transform-composition-utils

Represent and compose 2D affine transformations (translation, rotation, scaling, shear) as a single immutable matrix with left-to-right chaining.

```python
import math
from affine_transform_composition_utils import (
    Affine2D,
    IDENTITY,
    translation,
    rotation,
    scaling,
    shear,
)

# Build a transform by chaining in application order.
t = translation(10.0, 0.0).then(rotation(math.pi / 4)).then(scaling(2.0))

# Apply it to a point.
x, y = t.apply(1.0, 0.0)

# Compose several at once (each applied after the previous).
squashed = IDENTITY.compose(
    shear(shx=0.5),
    translation(-3.0, 1.0),
)

# Invert it, then round-trip back to identity.
back = t.inverse()
assert t.then(back).is_identity(epsilon=1e-9)
```

## Why this exists

The problem is ordering. When you build a transform out of pieces, the
question "does translate-then-rotate put the rotation about the origin or
about the translated point?" has exactly one right answer per call site, and
matrix conventions make it easy to get backwards. This library fixes the
convention: `a.then(b)` applies `a` first and `b` second, so chained code
reads in the order the operations happen. Internally that is the matrix
product `b @ a`, which is the part people usually get wrong when writing it
by hand.

The trade-off is that we store only the six meaningful coefficients of the
3x3 affine matrix and expose them as an immutable value object. That keeps
composition cheap and makes equality and hashing well-defined, but it means
there is no in-place mutation API. If you need to update a transform in a
loop, rebind the variable; the objects are small and cheap to allocate.

## Exported names

- `Affine2D` — the transform type. Construct with `Affine2D(a, b, c, d, tx, ty)` or `Affine2D.from_sequence(...)`.
- `IDENTITY` — a shared identity transform.
- `translation(tx, ty)`
- `rotation(radians)` — counter-clockwise about the origin.
- `scaling(sx, sy=None)` — uniform when `sy` is omitted.
- `shear(shx=0.0, shy=0.0)`

On `Affine2D`: `then`, `compose`, `apply`, `apply_many`, `determinant`,
`is_identity`, `almost_equals`, `inverse`, `as_tuple`, `as_matrix3`.

## The awkward edge

Floating-point composition accumulates error. After a few chained rotations,
`t.then(t.inverse())` will not compare equal with `==`; use
`is_identity(epsilon=...)` or `almost_equals(other, epsilon=...)`. The default
`EPSILON` is `1e-9`, which is fine for a handful of compositions but too tight
for long chains. `inverse()` raises `ZeroDivisionError` for a singular
transform rather than returning a garbage matrix, so check the determinant if
you are not sure the transform is invertible.
