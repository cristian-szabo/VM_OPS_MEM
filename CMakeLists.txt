cmake_minimum_required(VERSION 3.22)

project(Workspace)

list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

include(GlobalSettings)

add_subdirectory(VmOpsMem)
