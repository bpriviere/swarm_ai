#!/bin/bash

# Build the cpp files
#rm -r buildRelease

mkdir buildRelease
cd buildRelease
cmake -DCMAKE_BUILD_TYPE=Release ..
make

# Test it all works
cd ..

clear
echo
echo ===========================
echo  Testing Double Integrator
echo ===========================
python3 test_python_binding.py

clear
echo
echo ===========================
echo  Testing Single Integrator
echo ===========================
python3 test_python_binding_si.py

clear
echo
echo ===========================
echo     Testing Dubins 2D
echo ===========================
python3 test_python_binding_dubins2D.py
