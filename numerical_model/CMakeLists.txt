cmake_minimum_required(VERSION 3.14)
project(HoloMesons)

# 1. Базовые настройки компилятора
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Генерируем compile_commands.json для работы IntelliSense в VS Code
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 2. Настройка скачивания зависимостей
include(FetchContent)

# ВАЖНО: Правильные флаги для отключения лишнего.
# Имя флага для питона в autodiff — AUTODIFF_BUILD_PYTHON.
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_DOCS OFF CACHE BOOL "" FORCE)

# 3. Скачиваем Eigen3
message(STATUS "Downloading Eigen3...")
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(eigen)

# 4. Скачиваем autodiff
message(STATUS "Downloading autodiff...")
FetchContent_Declare(
    autodiff
    GIT_REPOSITORY https://github.com/autodiff/autodiff.git
    GIT_TAG        v1.1.2
)
FetchContent_MakeAvailable(autodiff)

message(STATUS "Downloading pybind11...")
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6 # Рекомендуется указывать конкретный стабильный релизный тег
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(meson_lib
    bind.cpp
    function_for_emulator.hpp
)

set(TARGET_PYTHON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../emulator")

set_target_properties(meson_lib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${TARGET_PYTHON_DIR}")

target_include_directories(meson_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(meson_lib PRIVATE 
    Eigen3::Eigen 
    autodiff::autodiff
)

add_executable(HoloMesons 
    testing_heavy_mesons.cpp
)

# Указываем, где лежат твои заголовочные файлы (.hpp)
target_include_directories(HoloMesons PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Подключаем скачанные библиотеки к твоему исполняемому файлу
target_link_libraries(HoloMesons PRIVATE Eigen3::Eigen autodiff::autodiff)

