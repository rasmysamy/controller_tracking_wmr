cmake_minimum_required(VERSION 3.0)
project(controller_tracking)

set(CMAKE_CXX_STANDARD 20)

# set O3 optimization and enable AVX
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx -mavx2 -mfma -fopenmp")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fopenmp")

# use OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# use Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})


# use Boost
find_package(Boost REQUIRED COMPONENTS system)
include_directories(${Boost_INCLUDE_DIRS})

# Use nlohmann json
find_package(nlohmann_json REQUIRED)



add_executable(controller_tracking main.cpp
        p3p/p3p.h
        p3p/solve_cubic.h
        p3p/refine_lambda.h
        p3p/matrix.h
        p3p/solve_eig0.h
        hungarian.h)

target_link_libraries(controller_tracking ${OpenCV_LIBS} ${Boost_LIBRARIES} nlohmann_json::nlohmann_json)
