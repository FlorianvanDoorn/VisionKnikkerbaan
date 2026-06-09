cmake_minimum_required(VERSION 3.16)
project(VisionKnikkerbaan VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
if(POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
endif()
find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW REQUIRED glfw3)
pkg_check_modules(IMGUI REQUIRED imgui)

find_path(IMGUI_BACKENDS_INCLUDE_DIR
    imgui_impl_glfw.h
    PATHS
        /usr/include/imgui/backends
        /usr/local/include/imgui/backends
)
find_file(IMGUI_IMPL_GLFW_SRC
    imgui_impl_glfw.cpp
    PATHS
        /usr/share/doc/libimgui-dev/examples/backends
        /usr/share/doc/imgui/examples/backends
)
find_file(IMGUI_IMPL_OPENGL3_SRC
    imgui_impl_opengl3.cpp
    PATHS
        /usr/share/doc/libimgui-dev/examples/backends
        /usr/share/doc/imgui/examples/backends
)

if(NOT IMGUI_BACKENDS_INCLUDE_DIR)
    message(FATAL_ERROR "Could not locate ImGui backend headers. Install libimgui-dev or add an external/imgui/backends folder.")
endif()
if(NOT IMGUI_IMPL_GLFW_SRC OR NOT IMGUI_IMPL_OPENGL3_SRC)
    message(FATAL_ERROR "Could not locate ImGui backend source files. Install libimgui-dev or add imgui_impl_glfw.cpp and imgui_impl_opengl3.cpp.")
endif()

add_executable(main
    src/VisionKnikkerbaan.cpp
    ${IMGUI_IMPL_GLFW_SRC}
    ${IMGUI_IMPL_OPENGL3_SRC}
)

target_include_directories(main PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${IMGUI_INCLUDE_DIRS}
    ${IMGUI_BACKENDS_INCLUDE_DIR}
    ${GLFW_INCLUDE_DIRS}
)

target_link_libraries(main PRIVATE
    ${OpenCV_LIBS}
    OpenGL::GL
    ${GLFW_LIBRARIES}
    ${IMGUI_LIBRARIES}
)

