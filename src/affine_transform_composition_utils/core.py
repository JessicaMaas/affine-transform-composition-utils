"""Core affine transform representation and composition.

A 2D affine transform is stored as a 3x3 matrix in row-major order::

    [ a  b  tx ]
    [ c  d  ty ]
    [ 0  0  1  ]

We store the six meaningful values (``a, b, c, d, tx, ty``) and treat the
bottom row as implicit.  Composition is matrix multiplication, so applying
transform ``B`` after transform ``A`` means computing ``B @ A``.  We expose
that through :meth:`Affine2D.then`, which reads left-to-right in
application order.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

# Tolerance used when comparing floats.  Consumers can rely on this being
# small enough for UI/geometry work while still absorbing the rounding that
# accumulates after a few compositions.
EPSILON = 1e-9


class Affine2D:
    """An immutable 2D affine transform.

    The six stored coefficients are ``a, b, c, d, tx, ty``.  Points are
treated as column vectors, so applying the transform to ``(x, y)`` gives::

        x' = a*x + b*y + tx
        y' = c*x + d*y + ty
    """

    __slots__ = ("a", "b", "c", "d", "tx", "ty")

    def __init__(self, a, b, c, d, tx, ty):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.tx = float(tx)
        self.ty = float(ty)

    # -- constructors -------------------------------------------------

    @classmethod
    def identity(cls) -> "Affine2D":
        """Return the do-nothing transform."""
        return cls(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    @classmethod
    def from_sequence(cls, values: Iterable[float]) -> "Affine2D":
        """Build from an iterable of exactly six coefficients.

        The order is ``(a, b, c, d, tx, ty)`` to match the storage order.
        """
        items = tuple(float(v) for v in values)
        if len(items) != 6:
            raise ValueError(
                "expected 6 coefficients (a, b, c, d, tx, ty), got %d" % len(items)
            )
        return cls(*items)

    # -- composition --------------------------------------------------

    def then(self, other: "Affine2D") -> "Affine2D":
        """Return ``other`` applied after ``self``.

        ``self.then(other)`` first applies ``self`` to a point and then
        applies ``other`` to the result.  This is the order that reads
        naturally in chained builder code.
        """
        if not isinstance(other, Affine2D):
            raise TypeError("then() requires an Affine2D")
        # Matrix product other @ self, expanded for the implicit bottom row.
        a = other.a * self.a + other.b * self.c
        b = other.a * self.b + other.b * self.d
        c = other.c * self.a + other.d * self.c
        d = other.c * self.b + other.d * self.d
        tx = other.a * self.tx + other.b * self.ty + other.tx
        ty = other.c * self.tx + other.d * self.ty + other.ty
        return Affine2D(a, b, c, d, tx, ty)

    def compose(self, *others: "Affine2D") -> "Affine2D":
        """Apply each transform in ``others`` in order, after ``self``."""
        result = self
        for other in others:
            result = result.then(other)
        return result

    # -- application --------------------------------------------------

    def apply(self, x: float, y: float) -> Tuple[float, float]:
        """Map the point ``(x, y)`` through this transform."""
        return (
            self.a * x + self.b * y + self.tx,
            self.c * x + self.d * y + self.ty,
        )

    def apply_many(self, points: Iterable[Tuple[float, float]]):
        """Map an iterable of points, returning a list of tuples."""
        return [self.apply(x, y) for (x, y) in points]

    # -- introspection ------------------------------------------------

    def determinant(self) -> float:
        """Return the linear part's determinant.

        A zero determinant means the transform collapses the plane, which
        also means :meth:`inverse` will fail.
        """
        return self.a * self.d - self.b * self.c

    def is_identity(self, epsilon: float = EPSILON) -> bool:
        """Return True when every coefficient is within ``epsilon`` of identity."""
        return (
            abs(self.a - 1.0) <= epsilon
            and abs(self.b) <= epsilon
            and abs(self.c) <= epsilon
            and abs(self.d - 1.0) <= epsilon
            and abs(self.tx) <= epsilon
            and abs(self.ty) <= epsilon
        )

    def almost_equals(self, other: "Affine2D", epsilon: float = EPSILON) -> bool:
        """Compare coefficients with a tolerance rather than ``==``."""
        if not isinstance(other, Affine2D):
            return NotImplemented
        return (
            abs(self.a - other.a) <= epsilon
            and abs(self.b - other.b) <= epsilon
            and abs(self.c - other.c) <= epsilon
            and abs(self.d - other.d) <= epsilon
            and abs(self.tx - other.tx) <= epsilon
            and abs(self.ty - other.ty) <= epsilon
        )

    def inverse(self, epsilon: float = EPSILON) -> "Affine2D":
        """Return the inverse transform.

        Raises ``ZeroDivisionError`` when the linear part is singular (the
        determinant is within ``epsilon`` of zero).  We deliberately do not
        return a sentinel value: a silently wrong inverse is worse than an
        exception at the call site.
        """
        det = self.determinant()
        if abs(det) <= epsilon:
            raise ZeroDivisionError("transform is not invertible (determinant ~ 0)")
        inv_det = 1.0 / det
        a = self.d * inv_det
        b = -self.b * inv_det
        c = -self.c * inv_det
        d = self.a * inv_det
        tx = -(a * self.tx + b * self.ty)
        ty = -(c * self.tx + d * self.ty)
        return Affine2D(a, b, c, d, tx, ty)

    def as_tuple(self) -> Tuple[float, float, float, float, float, float]:
        """Return ``(a, b, c, d, tx, ty)``."""
        return (self.a, self.b, self.c, self.d, self.tx, self.ty)

    def as_matrix3(self):
        """Return the full 3x3 matrix as a tuple of three row tuples."""
        return (
            (self.a, self.b, self.tx),
            (self.c, self.d, self.ty),
            (0.0, 0.0, 1.0),
        )

    # -- dunder -------------------------------------------------------

    def __eq__(self, other) -> bool:
        if not isinstance(other, Affine2D):
            return NotImplemented
        return self.as_tuple() == other.as_tuple()

    def __hash__(self) -> int:
        return hash(self.as_tuple())

    def __repr__(self) -> str:
        return (
            "Affine2D(a=%r, b=%r, c=%r, d=%r, tx=%r, ty=%r)"
            % (self.a, self.b, self.c, self.d, self.tx, self.ty)
        )


# A shared identity instance.  Affine2D is immutable in practice (no
# mutating methods are exposed), so sharing one instance is safe.
IDENTITY = Affine2D.identity()


# -- builders ---------------------------------------------------------

def translation(tx: float, ty: float) -> Affine2D:
    """Return a pure translation by ``(tx, ty)``."""
    return Affine2D(1.0, 0.0, 0.0, 1.0, tx, ty)


def rotation(radians: float) -> Affine2D:
    """Return a counter-clockwise rotation about the origin.

    Angles are in radians to match :mod:`math`; callers who think in degrees
    can use ``math.radians`` at the call site.
    """
    cos = math.cos(radians)
    sin = math.sin(radians)
    return Affine2D(cos, -sin, sin, cos, 0.0, 0.0)


def scaling(sx: float, sy: float = None) -> Affine2D:
    """Return a scaling transform.

    ``scaling(s)`` scales uniformly by ``s``; ``scaling(sx, sy)`` scales the
    axes independently.
    """
    if sy is None:
        sy = sx
    return Affine2D(sx, 0.0, 0.0, sy, 0.0, 0.0)


def shear(shx: float = 0.0, shy: float = 0.0) -> Affine2D:
    """Return a shear transform.

    ``shx`` shears the x axis by y (``x' = x + shx*y``) and ``shy`` shears
    the y axis by x (``y' = y + shy*x``).
    """
    return Affine2D(1.0, shx, shy, 1.0, 0.0, 0.0)
