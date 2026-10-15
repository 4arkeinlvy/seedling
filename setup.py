"""Builds the C core into the wheel as seedling/_libseedling*.so (loaded with ctypes, not
imported). Portable flags here; `make` builds a -march=native copy for local development."""

from pathlib import Path

from setuptools import Extension, setup

sources = sorted(str(p) for p in Path("csrc").glob("*.c"))

setup(
    ext_modules=[
        Extension(
            "seedling._libseedling",
            sources=sources,
            include_dirs=["csrc"],
            extra_compile_args=["-O3", "-std=gnu17", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        )
    ]
)
