cmake_minimum_required(VERSION 3.16)
project(test)

set(CMAKE_CXX_STANDARD 14)

add_executable(test main.cpp LR_model.cpp)
#add_executable(test tes.cpp)
#项目名称/工程名
#设置TENSORFLOW_DIR变量，变量内容为安装的tensorflow文件夹路径
set(TENSORFLOW_DIR /home/hqf/Downloads/tensorflow-1.13.1)
# 将源码目录保存到变量中
#add_library(demo ${SRC_LIST})   #明确指定包含哪些源文件
#设置包含的目录，项目中的include路径，换成自己的路径即可
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        /home/hqf/Downloads/tensorflow-1.13.1
        /home/hqf/Downloads/tensorflow-1.13.1/bazel-genfiles
        /home/hqf/Downloads/tensorflow-1.13.1/bazel-bin/tensorflow
        /home/hqf/Downloads/tensorflow-1.13.1/tensorflow/contrib/makefile/downloads/nsync/public
        /home/hqf/Downloads/tensorflow-1.13.1/tensorflow/contrib/makefile/downloads/eigen
        /home/hqf/Downloads/tensorflow-1.13.1/tensorflow/contrib/makefile/downloads/absl
)
#设置链接库搜索目录，项目中lib路径
link_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11)
link_directories(${TENSORFLOW_DIR}/home/hqf/Downloads/tensorflow-1.13.1/bazel-bin/tensorflow)  #动态链接库目录
#添加要编译的可执行文件
#设置 target 需要链接的库
#添加可执行文件所需要的库,连接libtensorflow_cc.so和libtensorflow_framework库，链接动态链接库
#target_link_libraries(demo tensorflow_cc tensorflow_framework)
target_link_libraries(test /home/hqf/Downloads/tensorflow-1.13.1/bazel-bin/tensorflow/libtensorflow_cc.so /home/hqf/Downloads/tensorflow-1.13.1/bazel-bin/tensorflow/libtensorflow_framework.so)