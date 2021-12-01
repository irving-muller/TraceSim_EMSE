from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "cython_mod.cmp_func", ["cython_mod/cmp_func.pyx"],
    ),
    Extension(
        "cython_mod.util.comparator", ["cython_mod/util/comparator.pyx"],
    ),

    Extension(
        "cython_mod.util.cython_util", ["cython_mod/util/cython_util.pyx"],
    ),
    Extension(
        "cython_mod.method.method", ["cython_mod/method/method.pyx"],
    ),
    Extension(
        "cython_mod.method.pdm", ["cython_mod/method/pdm.pyx"],
    ),
    Extension(
        "cython_mod.method.edit_distance", ["cython_mod/method/edit_distance.pyx"],
    ),
    Extension(
        "cython_mod.method.prefix_match", ["cython_mod/method/prefix_match.pyx"],
    ),
    Extension(
        "cython_mod.method.crash_graph", ["cython_mod/method/crash_graph.pyx"],
    ),
    Extension(
        "cython_mod.method.bartz_08",
        ["cython_mod/method/bartz_08.pyx"],
    ),
    Extension(
        "cython_mod.method.brodie_05",
        ["cython_mod/method/brodie_05.pyx"],
    ),
    Extension(
        "cython_mod.method.trace_sim",
        ["cython_mod/method/trace_sim.pyx"],
    ),

    # BOW
    Extension(
        "cython_mod.bow_method.bow_method", ["cython_mod/bow_method/bow_method.pyx"],
    ),
    Extension(
        "cython_mod.bow_method.durfex", ["cython_mod/bow_method/durfex.pyx"],
    ),
    Extension(
        "cython_mod.cmp_func_bow", ["cython_mod/cmp_func_bow.pyx"],
    ),

]

# setup(ext_modules= cythonize(['cython_mod/*.pyx'], annotate=True,),include_dirs=[numpy.get_include()])


setup(ext_modules=cythonize(ext_modules, annotate=True, ), include_dirs=[numpy.get_include(), 'cython_mod/opt_algorithm/cpp'], language='c++')
# setup(ext_modules= cythonize(['pdm.pyx', 'edit_distance.pyx', 'bartz_08.pyx', 'brodie_05.pyx', 'cython_util.pyx'], annotate=True,),include_dirs=[numpy.get_include()])

# from setuptools.extension import Extension
# ext = [Extension("brodie_05", ["brodie_05.pyx"], annotate=True, define_macros=[('CYTHON_TRACE', '1')])]
# setup(ext_modules=cythonize(ext), include_dirs=[numpy.get_include()])
