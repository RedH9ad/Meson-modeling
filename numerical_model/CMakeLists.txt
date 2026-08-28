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

# 5. Сборка твоего проекта
add_executable(HoloMesons 
    main.cpp
)

add_executable(kfit
    fitting_k.cpp
)

# Указываем, где лежат твои заголовочные файлы (.hpp)
target_include_directories(HoloMesons PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Подключаем скачанные библиотеки к твоему исполняемому файлу
target_link_libraries(HoloMesons PRIVATE Eigen3::Eigen autodiff::autodiff)

target_include_directories(kfit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Подключаем скачанные библиотеки к твоему исполняемому файлу
target_link_libraries(kfit PRIVATE Eigen3::Eigen autodiff::autodiff)