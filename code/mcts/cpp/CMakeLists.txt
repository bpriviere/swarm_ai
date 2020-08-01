cmake_minimum_required (VERSION 3.5)
project (mcts)

# Enable C++17 and warnings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# find_package(Boost 1.58 COMPONENTS program_options system REQUIRED)

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(test_mcts
  test_mcts.cpp
)
target_link_libraries(test_mcts
  Eigen3::Eigen
)
