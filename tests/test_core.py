import math
import unittest

from affine_transform_composition_utils import (
    Affine2D,
    IDENTITY,
    rotation,
    scaling,
    shear,
    translation,
)


class TestConstruction(unittest.TestCase):
    def test_identity_coefficients(self):
        self.assertEqual(IDENTITY.as_tuple(), (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        self.assertTrue(IDENTITY.is_identity())

    def test_from_sequence_round_trip(self):
        values = (2.0, 0.5, -0.25, 3.0, 7.0, -4.0)
        t = Affine2D.from_sequence(values)
        self.assertEqual(t.as_tuple(), values)

    def test_from_sequence_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            Affine2D.from_sequence([1.0, 0.0, 0.0, 1.0])

    def test_from_sequence_accepts_generator(self):
        t = Affine2D.from_sequence(float(i) for i in range(6))
        self.assertEqual(t.as_tuple(), (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))


class TestBuilders(unittest.TestCase):
    def test_translation_moves_point(self):
        t = translation(3.0, -2.0)
        self.assertEqual(t.apply(1.0, 1.0), (4.0, -1.0))

    def test_rotation_quarter_turn(self):
        r = rotation(math.pi / 2.0)
        x, y = r.apply(1.0, 0.0)
        self.assertTrue(math.isclose(x, 0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(y, 1.0, abs_tol=1e-12))

    def test_scaling_uniform_default(self):
        s = scaling(2.0)
        self.assertEqual(s.apply(3.0, 4.0), (6.0, 8.0))

    def test_scaling_independent_axes(self):
        s = scaling(2.0, 3.0)
        self.assertEqual(s.apply(1.0, 1.0), (2.0, 3.0))

    def test_shear_x_axis(self):
        sh = shear(shx=1.0)
        # x' = x + 1*y, y' = y
        self.assertEqual(sh.apply(0.0, 2.0), (2.0, 2.0))

    def test_shear_y_axis(self):
        sh = shear(shy=1.0)
        self.assertEqual(sh.apply(2.0, 0.0), (2.0, 2.0))


class TestComposition(unittest.TestCase):
    def test_then_applies_in_order(self):
        # Translate then rotate: rotation about the origin happens after the
        # point has been moved, so the result is not the same as rotating a
        # point at the origin.
        t = translation(1.0, 0.0)
        r = rotation(math.pi / 2.0)
        combined = t.then(r)
        x, y = combined.apply(0.0, 0.0)
        self.assertTrue(math.isclose(x, 0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(y, 1.0, abs_tol=1e-12))

    def test_then_is_not_commutative(self):
        t = translation(1.0, 0.0)
        r = rotation(math.pi / 2.0)
        a = t.then(r).apply(0.0, 0.0)
        b = r.then(t).apply(0.0, 0.0)
        self.assertFalse(
            math.isclose(a[0], b[0], abs_tol=1e-12)
            and math.isclose(a[1], b[1], abs_tol=1e-12)
        )

    def test_compose_variadic(self):
        combined = IDENTITY.compose(
            translation(1.0, 0.0),
            scaling(2.0, 2.0),
            translation(0.0, 1.0),
        )
        # (0,0) -> (1,0) -> (2,0) -> (2,1)
        self.assertEqual(combined.apply(0.0, 0.0), (2.0, 1.0))

    def test_compose_no_args_returns_self(self):
        t = translation(5.0, 5.0)
        self.assertTrue(t.compose().almost_equals(t))

    def test_then_rejects_non_transform(self):
        with self.assertRaises(TypeError):
            IDENTITY.then("nope")

    def test_identity_is_neutral(self):
        t = translation(2.0, 3.0).then(rotation(0.3)).then(scaling(1.5))
        self.assertTrue(t.then(IDENTITY).almost_equals(t))
        self.assertTrue(IDENTITY.then(t).almost_equals(t))


class TestInverse(unittest.TestCase):
    def test_inverse_round_trip(self):
        t = translation(2.0, -1.0).then(rotation(0.7)).then(scaling(3.0, 2.0))
        inv = t.inverse()
        round_tripped = t.then(inv)
        self.assertTrue(round_tripped.is_identity(epsilon=1e-9))

    def test_inverse_of_singular_raises(self):
        singular = scaling(0.0, 1.0)
        with self.assertRaises(ZeroDivisionError):
            singular.inverse()

    def test_inverse_applies_to_points(self):
        t = translation(4.0, 5.0)
        inv = t.inverse()
        self.assertEqual(inv.apply(4.0, 5.0), (0.0, 0.0))


class TestIntrospection(unittest.TestCase):
    def test_determinant_of_scaling(self):
        self.assertTrue(math.isclose(scaling(3.0, 4.0).determinant(), 12.0))

    def test_determinant_of_rotation_is_one(self):
        self.assertTrue(math.isclose(rotation(1.234).determinant(), 1.0, abs_tol=1e-12))

    def test_almost_equals_tolerance(self):
        a = Affine2D(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        b = Affine2D(1.0 + 1e-12, 0.0, 0.0, 1.0, 0.0, 0.0)
        self.assertTrue(a.almost_equals(b))
        self.assertFalse(a.almost_equals(b, epsilon=1e-15))

    def test_almost_equals_non_transform(self):
        self.assertIs(IDENTITY.almost_equals(3), NotImplemented)

    def test_as_matrix3_shape(self):
        t = translation(1.0, 2.0)
        self.assertEqual(
            t.as_matrix3(),
            ((1.0, 0.0, 1.0), (0.0, 1.0, 2.0), (0.0, 0.0, 1.0)),
        )

    def test_equality_and_hash(self):
        a = translation(1.0, 2.0)
        b = translation(1.0, 2.0)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)

    def test_equality_with_other_type(self):
        self.assertNotEqual(IDENTITY, "identity")

    def test_apply_many(self):
        t = translation(1.0, 1.0)
        self.assertEqual(
            t.apply_many([(0.0, 0.0), (1.0, 2.0)]),
            [(1.0, 1.0), (2.0, 3.0)],
        )


class TestCompositionIsAssociative(unittest.TestCase):
    def test_associativity(self):
        a = translation(1.0, 2.0)
        b = rotation(0.5)
        c = scaling(2.0, 3.0)
        left = a.then(b).then(c)
        right = a.then(b.then(c))
        self.assertTrue(left.almost_equals(right))


if __name__ == "__main__":
    unittest.main()
