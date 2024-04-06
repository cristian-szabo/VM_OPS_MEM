cmake_minimum_required(VERSION 3.22)

include(GenerateExportHeader)

project(VmOpsMem LANGUAGES CXX)

set(PROJECT_FILES
    vm_ops_mem.cpp
)

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
    list(APPEND PROJECT_FILES
        ops_intel_64.cpp
    )
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    list(APPEND PROJECT_FILES
        ops_arm_64.cpp
    )
else()
    message(FATAL_ERROR "Arch not supported!")
endif()

add_library(${PROJECT_NAME} SHARED ${PROJECT_FILES})

generate_export_header(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
    target_compile_options(${PROJECT_NAME} PRIVATE -mamx-int8 -mamx-bf16 -mamx-tile -mavxvnni)
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    target_compile_options(${PROJECT_NAME} PRIVATE -march=armv8.4-a+bf16+i8mm+dotprod+fp16)
endif()

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
