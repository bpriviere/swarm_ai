cmake_minimum_required (VERSION 3.5)
project (mcts)

# Enable C++14 and warnings
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Boost 1.58 COMPONENTS program_options system REQUIRED)
find_package(PkgConfig)
pkg_check_modules(YamlCpp yaml-cpp)

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(swarmgame
  swarmgame.cpp
)
target_link_libraries(swarmgame
  ${Boost_LIBRARIES}
  yaml-cpp
)

add_executable(test_glas
  test_glas.cpp
)
target_link_libraries(test_glas
  ${Boost_LIBRARIES}
  yaml-cpp
)

add_executable(test_tictactoe
  test_tictactoe.cpp
)
