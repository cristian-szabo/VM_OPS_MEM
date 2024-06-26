cmake_minimum_required(VERSION 3.22)

include(GNUInstallDirs)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_INSTALL_BINDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_LIBDIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_INSTALL_LIBDIR})

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

set(Python_FIND_VIRTUALENV FIRST)

message(STATUS "TARGET SYSTEM NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "TARGET SYSTEM ARCH: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "TARGET SYSTEM VERSION: ${CMAKE_SYSTEM_VERSION}")
message(STATUS "HOST SYSTEM NAME: ${CMAKE_HOST_SYSTEM_NAME}")
message(STATUS "HOST SYSTEM ARCH: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "HOST SYSTEM VERSION: ${CMAKE_HOST_SYSTEM_VERSION}")
