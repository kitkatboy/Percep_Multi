cmake_minimum_required(VERSION 2.8.4)
project(Reseau_neurones)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Include OpenCV library
find_package(
    OpenCV REQUIRED
)

include_directories(
    include
)

file(
    GLOB_RECURSE
    SOURCE_FILES
    main.cpp
    src/*
)

add_executable(
    Reseau_neurones
    ${SOURCE_FILES}
)

#Configuration de l'édition de liens
target_link_libraries(
    Reseau_neurones
    ${OpenCV_LIBS}
)