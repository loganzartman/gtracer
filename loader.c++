#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include "Vec3.hh"
#include "Tri.hh"
#include "Geometry.hh"
#include "loader.hh"

using namespace std;

const vector<Float3>& load(string filename, vector<Float3>& vertices, float scale) {
    ifstream file(filename);
    string s;

    if (!file.is_open()) {
        cerr << "Error opening file" << endl;
        exit(0);
    }

    string command;
    while (!file.eof()) {
        file >> command;
        if (command == "v") {
            float one, two, three;
            file >> one >> two >> three;
            Float3 vertex(one, two, three);
            vertices.push_back(vertex * scale);
        } else {
            // noop
        }
    }

    return vertices;
}

const vector<Geometry*>& triangulate(string filename, vector<Float3> vertices, vector<Geometry*>& objs, const Material* mat) {
    ifstream file(filename); 
    string s;

    if (!file.is_open()) {
        cerr << "Error opening file" << endl;
        exit(0);
    }

    while (!file.eof()) {
        getline(file, s);
        if (s[0] == 'f')
            break;
    }
    
    int i = 0;
    while(s[0] == 'f') {
        while(s[i] == 'f' || s[i] == ' ')
            ++i;

        int j = i, k = i;
        while (s[i] != ' ')
            k = ++i;

        int one = stof(s.substr(j, k - j));

        ++i;
        j = i, k = i;
        while(s[i] != ' ')
            k = ++i;

        int two = stof(s.substr(j, k - j));
 
        ++i;
        j = i, k = i;
        while(s[i] != ' ' && static_cast<size_t>(i) != s.length())
            k = ++i;

        int three = stof(s.substr(j, k - j));

        Geometry *obj = new Tri(vertices[one-1], vertices[two-1], vertices[three-1], mat);
        objs.push_back(obj);

        getline(file, s);
        i = 0;
    }
    return objs;
}
