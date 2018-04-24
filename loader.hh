#ifndef LOADER_HH
#define LOADER_HH

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include "Vec3.hh"
#include "Geometry.hh"

std::vector<float3> load(std::string filename);
const std::vector<Geometry*>& triangulate(std::string filename, std::vector<float3> vertices);

#endif
