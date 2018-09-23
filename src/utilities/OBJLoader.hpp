#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include "floats.hpp"
#include "geometry.hpp"

std::vector<Mesh> loadWavefront(std::string const srcFile, int rank, int root,bool quiet = true);