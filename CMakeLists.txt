cmake_minimum_required(VERSION 3.18)

project(CUDAProject LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设置 UTF-8 编码
if(MSVC)
    # 对于 C++ 文件
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    # 对于 CUDA 文件，通过 -Xcompiler 传递给主机编译器
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

# 查找 CUDA
find_package(CUDAToolkit REQUIRED)

# 添加可执行文件
add_executable(error_hand "Week1_CUDA入门/06_matrix_add.cu")

# 设置 CUDA 架构（RTX 4060 使用 Ada Lovelace 架构 sm_89）
set_target_properties(error_hand PROPERTIES
    CUDA_ARCHITECTURES "89"
)

# Windows 特定设置：禁用 manifest 嵌入
if(MSVC)
    set_target_properties(error_hand PROPERTIES
        LINK_FLAGS "/MANIFEST:NO"
    )
endif()

# 链接 CUDA 库
target_link_libraries(error_hand CUDA::cudart)
