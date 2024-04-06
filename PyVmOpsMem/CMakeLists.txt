cmake_minimum_required(VERSION 3.22)

project(PyVmOpsMem LANGUAGES CXX)

set(PROJECT_FILES
    mem.h
    mem.cpp
    vm_ops_mem.h
    vm_ops_mem.cpp
)

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
    list(APPEND PROJECT_FILES
        ops_x86_64.h
        ops_x86_64.cpp
    )
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    list(APPEND PROJECT_FILES
        ops_arm_64.h
        ops_arm_64.cpp
    )
else()
    message(FATAL_ERROR "Arch not supported!")
endif()

pybind11_add_module(${PROJECT_NAME} MODULE ${PROJECT_FILES})

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
    target_compile_options(${PROJECT_NAME} PRIVATE -mamx-int8 -mamx-bf16 -mamx-tile -mavxvnni)
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    target_compile_options(${PROJECT_NAME} PRIVATE -march=armv8.2-a+i8mm+bf16+dotprod)
endif()

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
