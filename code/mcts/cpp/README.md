## C++ Version

### Build

```
mkdir buildRelease
cd buildRelease
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

* Use `buildDebug`, `CMAKE_BUILD_TYPE=Debug`, and QtCreator or Clion to debug

### Run
```
./test_swarmgame && python3 ../plot.py
```

### Notes

* MCTS is generic in the style of libMultiRobotPlanning, with templated State, Action, and GameLogic
* By default, ties are not broken randomly, as in the Python version
* The GameState is templated by #Attackers/#Defenders. This might be annoying for python bindings as we need to know at compile time what variant to run. This design choice was made to have improved static allocation.

### Todo

* Profile and perf improvements
  * Splitting the gamestate in A/B could avoid some copy operations
  * If we have an upper bound on the branching, we can use static allocation for children pointers and actions
* Multi-Robot extension (needs cartesian product, everything else is prepared)