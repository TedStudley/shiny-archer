shiny-archer
============

Python incompressible flow advection and diffusion simulation using the FEniCS Finite Element library

This program uses the FEniCS Finite Element library. This library can be found at fenicsproject.org. To download using apt-get, run the following commands in terminal: 

    `% sudo add-apt-repository ppa:fenics-packages/fenics`
    `% sudo apt-get update`
    `% sudo apt-get install fenics`
    `% sudo apt-get dist-upgrade`

FEniCS will then be installed on the machine, and can be accessed in Python by using
    
    `from dolfin import *`

or in C++ by using

    `#include <dolfin.h>`

Any of the scripts from this repository can be run through Python, for example:

    `% python archer1d.py`
