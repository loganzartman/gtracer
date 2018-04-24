#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include "Vec3.hh"

using namespace std;

vector<float3> load(string filename) {
    ifstream file(filename);
    vector<float3> vertices;
    string s;

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

        vertex.x = stof(s.substr(j, k-j));

        while (s[i] == ' ')
            ++i;

        int q = i, w = i;
        while (s[i] != ' ')
            w = ++i;

        vertex.y = stof(s.substr(q, w-q));
        
        while (s[i] == ' ')
            ++i;

        int a = i, b = i;
        while (s[i] != ' ' && static_cast<size_t>(i) != s.length())
            b = ++i;

        vertex.z = stof(s.substr(a, b-a));
        
        vertices.push_back(vertex);
        ++v;
        getline(file, s);
    }
    return vertices;
}
