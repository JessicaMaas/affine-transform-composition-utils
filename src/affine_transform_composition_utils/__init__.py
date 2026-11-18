"""Compose 2D affine transformations as a single matrix.

This package exposes a small, dependency-free toolkit for building and
chaining translation, rotation, scaling, and shear transforms.
"""

from .core import Affine2D, IDENTITY, translation, rotation, scaling, shear

__all__ = [
    "Affine2D",
    "IDENTITY",
    "translation",
    "rotation",
    "scaling",
    "shear",
]
