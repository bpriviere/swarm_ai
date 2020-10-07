#!/usr/bin/env bash


build_cpp() {
    # Build cpp files
    echo "Building cpp files"   

    mkdir buildRelease
    cd buildRelease
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

    cd ..
}

test_bindings() {
    # Test bindings
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

}

# Main program to run the stuff
main () {
    build_cpp
    test_bindings
    
}

main "$@"

