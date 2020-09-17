cmake_minimum_required (VERSION 3.5)
project (mcts)

# Enable C++14 and warnings
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

find_package(Eigen3 REQUIRED NO_MODULE)

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
)

# Python bindings

add_subdirectory(pybind11)

pybind11_add_module(mctscpp
  python_bindings.cpp
)

pybind11_add_module(mctscppsi
  python_bindings_si.cpp
)

pybind11_add_module(mctscppdubins2D
  python_bindings_dubins2D.cpp
)
