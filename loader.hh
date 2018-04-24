#ifndef LOADER_HH
#define LOADER_HH

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include "Vec3.hh"
#include "Geometry.hh"
#include "Material.hh"

const std::vector<Geometry*>& load(std::string filename, std::vector<Geometry*>& objs, float scale, const Material* mat);

#endif
