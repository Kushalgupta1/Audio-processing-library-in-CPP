// #include <bits/stdc++.h> 
#include <iostream>   
#include <vector>
#include <fstream>
#include "audio.h"

using namespace std;
typedef long long ll;

typedef vector<float> vf;

int main(int argc, char **argv)
{
    ios_base::sync_with_stdio(false);
    ifstream myin;
    ofstream myout;
    // obj is an object of Matrix math class
    // myin will be used as the ifstream and myout will be the ofstream

    // Taking input by various cases 
    if (argc == 3) 
    {
 
            myin.open(argv[1]);
            if (!myin)
            {
                cout << "Input matrix File not found \n";
                return 0;
            }
            myin.close();
			//calling libaudioAPI
			pred_t fake[3];
            pred_t* pred = libaudioAPI(argv[1], fake);

			vector<string> labels{ "_silence_",  "_unknown_", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"};

            myout.open(argv[2], ofstream::out | ofstream::app);
            myout << argv[1] << " " << labels[pred[0].label] << " " << labels[pred[1].label] << " " <<  labels[pred[2].label] << " " << pred[0].prob << " " << pred[1].prob << " " << pred[2].prob << "\n";

    }
    else
    {
        std::cout << "Incorrect number of arguments\n";
    }

    return 0;
}