## C++ Version

### Dependencies

```
sudo apt install -y libeigen3-dev libyaml-cpp-dev
```

This relies on pybind11, which is a submodule, so don't forget to

```
git submodule init 
git submodule update
```

### Build

```
mkdir buildRelease
cd buildRelease
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

* Use `buildDebug`, `CMAKE_BUILD_TYPE=Debug`, and QtCreator or Clion to debug

* if you use anaconda, you must link the python version to cmake:
```
cmake -DPYTHON_EXECUTABLE=path/to/python -DCMAKE_BUILD_TYPE=Release ..
```

### Run

In general, use `run.py` in the code folder to automatically use the Python bindings.

#### Testing Python Bindings

This uses the bindings in `buildRelease` and has some examples on how to use the bindings.

```
python3 test_python_binding.py
```

### Notes

* MCTS is generic in the style of libMultiRobotPlanning, with templated State, Action, and GameLogic
* By default, ties are not broken randomly, as in the Python version
* The GameState is no longer templated by #Attackers/#Defenders to allow Python bindings. The performance loss is minimal, since the runtime is dominated by rollouts, not by copy operations.
* The code is templated by dynamics model, requiring different bindings for each robot type

### Todo

* Profile and perf improvements
  * Splitting the gamestate in A/B could avoid some copy operations
  * If we have an upper bound on the branching, we can use static allocation for children pointers and actions

### Profiling

```
mkdir buildProfile
cd buildProfile
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make
cd ..
perf record --call-graph dwarf python3 test_python_binding.py
~/sw/hotspot-v1.2.0-x86_64.AppImage perf.data
```

Where hotspot is from https://www.kdab.com/hotspot-gui-linux-perf-profiler/

## CLANG

Using Clang might provide more useful error messages over the default gcc compiler.

```
mkdir buildClang
CC=clang CXX=clang++ cmake ..
```