#include <string>

using namespace std;

void load(string filename) {
    ifstream file;
    file.open(filename);
    
    vector<float3> vertices;

    string s;

    while (!file.eof()) {
        file >> s;
        if (s[0] == 'v')
            break;
    }

    int i;
    int v = 0;
    while (s[0] == 'v') {
        i = 0;
        float3 vertex;

        while(s[i] == ' ')
            ++i;
        i+=2;
        int j = i, k = i;
        while (s[i] != ' ')
            k = ++i;

        vertex.x = 0f;
        vertex.y = 0f;
        vertex.z = 0f;
    }
}
