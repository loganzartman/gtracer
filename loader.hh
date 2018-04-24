#ifndef LOADER_HH
#define LOADER_HH

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include "Vec3.hh"
#include "Geometry.hh"

const std::vector<Float3>& load(std::string filename, std::vector<Float3>& vertices);
const std::vector<Geometry*>& triangulate(std::string filename, std::vector<Float3> vertices, std::vector<Geometry*>& objs);

#endif
