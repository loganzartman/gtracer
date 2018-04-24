#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include "Vec3.hh"
#include "Tri.hh"
#include "Geometry.hh"
#include "loader.hh"

using namespace std;

const vector<float3>& load(string filename, vector<float3>& vertices) {
    ifstream file(filename);
    string s;

    if (!file.is_open()) {
        cerr << "Error opening file" << endl;
        exit(0);
    }

    while (!file.eof()) {
        getline(file, s);
        if (s[0] == 'v')
            break;
    }
    
    int i;
    int v = 0;
    while (s[0] == 'v') {
        i = 0;
        float3 vertex;

        while (s[i] == ' ')
            ++i;

        i+=2;

        int j = i, k = i;
        while (s[i] != ' ')
            k = ++i;

        vertex.x = stof(s.substr(j, k - j));

        while (s[i] == ' ')
            ++i;

        j = i, k = i;
        while (s[i] != ' ')
            k = ++i;

        vertex.y = stof(s.substr(j, k - j));
        
        while (s[i] == ' ')
            ++i;

        j = i, k = i;
        while (s[i] != ' ' && static_cast<size_t>(i) != s.length())
            k = ++i;

        vertex.z = stof(s.substr(j, k - j));
        
        vertices.push_back(vertex);
        ++v;
        getline(file, s);
    }
    return vertices;
}

const vector<Geometry*>& triangulate(string filename, vector<float3> vertices, vector<Geometry*>& objs) {
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

        Geometry *obj = new Tri(vertices[one], vertices[two], vertices[three]);
        objs.push_back(obj);
    }
    return objs;
}
