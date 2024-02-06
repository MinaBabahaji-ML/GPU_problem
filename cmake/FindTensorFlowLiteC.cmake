# When setting cmake_minimum_required to 3.12+, one can set TensorflowLite_ROOT variable
# at configure time, and find_library will automatically look inside 
# ${TensorflowLite_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE}, where
# ${CMAKE_LIBRARY_ARCHITECTURE} is x86_64-linux-gnu for Linux,
# and aarch64-linux-android when cross compiling from Linux to Android

# Explanation of NO_CMAKE_FIND_ROOT_PATH: when cross compiling for android, 
# the CMAKE_FIND_ROOT_PATH variable is set to the root folder of the NDK by
# android.toolchain.cmake, without NO_CMAKE_FIND_ROOT_PATH here, find_path 
# and find_library will append ${CMAKE_FIND_ROOT_PATH} to every folder 
# it searches.
find_path(TensorFlowLiteC_INCLUDE_DIR
    NAMES
        tensorflow/lite/c/c_api.h
    NO_CMAKE_FIND_ROOT_PATH
    REQUIRED
)

find_library(TensorFlowLiteC_LIBRARY 
    NAMES            
        tensorflowlite_c
    NO_CMAKE_FIND_ROOT_PATH
    REQUIRED
)

find_library(TensorFlowLiteGPUDelegate_LIBRARY 
    NAMES            
        tensorflowlite_gpu_delegate
    NO_CMAKE_FIND_ROOT_PATH
    REQUIRED
)

set(TensorFlowLiteC_LIBRARIES
    ${TensorFlowLiteGPUDelegate_LIBRARY}
    ${TensorFlowLiteC_LIBRARY}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    TensorFlowLiteC
    FOUND_VAR TensorFlowLiteC_FOUND
    REQUIRED_VARS TensorFlowLiteC_LIBRARIES TensorFlowLiteC_INCLUDE_DIR
)
