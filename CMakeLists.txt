cmake_minimum_required(VERSION 3.22)

project(Workspace)

list(APPEND CMAKE_MODULE_PATH 
    ${CMAKE_CURRENT_SOURCE_DIR}/CMake)
    
include(CPM)
include(GlobalSettings)

CPMAddPackage(
    NAME pybind11
    VERSION 2.12.0
    GITHUB_REPOSITORY pybind/pybind11
)

add_subdirectory(PyVmOpsMem)
