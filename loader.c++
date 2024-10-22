#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "GeomData.hh"
#include "Geometry.hh"
#include "Sphere.hh"
#include "Tri.hh"
#include "Vec3.hh"
#include "loader.hh"

using namespace std;

const vector<Geometry>& load(string filename, vector<Geometry>& objs,
                             float scale, Float3 translate, const Material* mat) {
    ifstream file(filename);
    string s;

    if (!file.is_open()) {
        cerr << "Error opening file" << endl;
        exit(0);
    }

    vector<Float3> vertices;

    string command;
    while (!file.eof()) {
        file >> command;
        if (command == "#") {
            // comment line: ignore
            string _;
            getline(file, _);
        } else if (command == "v") {
            // vertex: v x y z
            float one, two, three;
            file >> one >> two >> three;
            Float3 vertex(one, two, three);
            vertices.push_back(vertex * scale + translate);
        } else if (command == "f") {
            // face: f v1 v2 v3
            string sone, stwo, sthree;
            file >> sone >> stwo >> sthree;

            // face syntax allows slash-separated texture and normal indices:
            // face: vertex1/tex1/norm1 v2/t2/n2 v3/t3/n3
            int one, two, three;
            istringstream ssone(sone), sstwo(stwo), ssthree(sthree);
            ssone >> one;
            sstwo >> two;
            ssthree >> three;

            // create triangle
            TriData tri = TriData{vertices[one - 1], vertices[two - 1],
                                  vertices[three - 1], 0};
            tri.n = Tri::normal(tri, vertices[one - 1], vertices[two - 1]);

            Geometry obj = Geometry(tri, mat);
            objs.push_back(obj);
        } else if (command == "sphere") {
            // sphere: sphere x y z r
            float one, two, three, r;
            file >> one >> two >> three >> r;

            // create sphere
            Geometry obj =
                Geometry(SphereData{Float3(one, two, three) * scale + translate, r * scale}, mat);
            objs.push_back(obj);
        } else {
            // noop
        }
    }

    return objs;
}
