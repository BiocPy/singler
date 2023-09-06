"""Setup file for singler. Use setup.cfg to configure your project.

This file was generated with PyScaffold 4.5.
PyScaffold helps you to put up the scaffold of your new Python project.
Learn more under: https://pyscaffold.org/
"""
from setuptools import setup, Extension
import assorthead
import mattress

if __name__ == "__main__":
    try:
        setup(
            use_scm_version={"version_scheme": "no-guess-dev"},
            ext_modules=[
                Extension(
                    "singler.core",
                    [
                        "src/singler/lib/Markers.cpp",
                        "src/singler/lib/bindings.cpp",
                        "src/singler/lib/find_classic_markers.cpp",
                    ],
                    include_dirs=[assorthead.includes()] + mattress.includes(),
                    language="c++",
                    extra_compile_args=[
                        "-std=c++17",
                    ],
                )
            ],
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
