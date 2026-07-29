"""AURA test suite.

A real package rather than a bare directory because several tests are imported as
modules by other tests (``aura.tests.test_mri_foundation`` supplies volume fixtures),
and because ``aura/tests`` and the engines' own module names would otherwise collide
in pytest's rootdir-based import mode.
"""
