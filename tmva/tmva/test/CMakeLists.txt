############################################################################
# CMakeLists.txt file for building ROOT TMVA tests.
# @author Simon Pfreundschuh
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

# Recursive addition of tests
ROOT_ADD_TEST_SUBDIRECTORY(DNN)

# For google tests
ROOT_ADD_GTEST(testTmvaOutputCompat TestTmvaOutputCompat.cxx LIBRARIES TMVA)
