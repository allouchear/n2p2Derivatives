n2p2 - The neural network potential package, version midifed by A.R. Allouche
=============================================================================

[![DOI](https://zenodo.org/badge/142296892.svg)](https://zenodo.org/badge/latestdoi/142296892)
[![Build Status](https://travis-ci.org/CompPhysVienna/n2p2.svg?branch=master)](https://travis-ci.org/CompPhysVienna/n2p2)
[![Coverage](https://codecov.io/gh/CompPhysVienna/n2p2/branch/master/graph/badge.svg)](https://codecov.io/gh/CompPhysVienna/n2p2)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository provides ready-to-use software for high-dimensional neural
network potentials in computational physics and chemistry.
Added by A.R. Allouche : 
 * NN for partial charges and dipoles
 * High derivatives of energies (2, 3, 4)
 * dftd3 grimme correction to energie

# Documentation

## Build your own documentation
It is also possible to build your own documentation for offline reading.
Install the above dependencies, change to the `src` directory and try to build
the documentation:
```
source env.sh
cd dftd3-lib-0.9
make
cd ..
cd src
make doc
```
If the build process succeeds you can browse through the documentation starting
from the main page in:
```
doc/html/index.html
```

# Authors

 - Andreas Singraber (University of Vienna)
 - Abdulrahman Allouche (Lyon 1 University)

# License

This software is licensed under the [GNU General Public License version 3 or any later version (GPL-3.0-or-later)](https://www.gnu.org/licenses/gpl.txt).
