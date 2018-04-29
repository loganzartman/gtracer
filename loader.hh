#ifndef LOADER_HH
#define LOADER_HH

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Geometry.hh"
#include "Material.hh"
#include "Vec3.hh"

const std::vector<Geometry>& load(std::string filename,
                                  std::vector<Geometry>& objs, float scale,
                                  const Material* mat);

#endif
