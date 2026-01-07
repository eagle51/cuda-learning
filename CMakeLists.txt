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
add_executable(vector_add Week1_CUDA入门/03_vector_add.cu)

# 设置 CUDA 架构（RTX 4060 使用 Ada Lovelace 架构 sm_89）
set_target_properties(vector_add PROPERTIES
    CUDA_ARCHITECTURES "89"
)

# 链接 CUDA 库
target_link_libraries(vector_add CUDA::cudart)
