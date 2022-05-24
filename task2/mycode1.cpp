// #include <bits/stdc++.h> 
#include "mklfullyconnected.h"
#include "Matrixmath.h"
#include <iostream>  
#include "mkl.h"
#include <vector>
#include <fstream>
#include <math.h>
#include <string.h>
#include <cassert>

using namespace std;
typedef long long ll;

typedef vector<float> vf;

int main(int argc, char **argv)
{
    ios_base::sync_with_stdio(false);
    ifstream myin;
    ofstream myout;
    Matrixmath obj;
    // obj is an object of Matrix math class
    // myin will be used as the ifstream and myout will be the ofstream

    // Taking input by various cases 
    if (argc > 1) 
    {
        if (strcmp(argv[1], "fullyconnected") == 0 )
        {

            try
            {
                if ((argc != 7) ||(strcmp(argv[2],"mkl")!=0))
                {
                    throw argc;
                }
            }
            catch (int z)
            {
                cout << "All arguments for fullyconnected not provided\n";
                return 0;
            }

            // assert(argc==6);
            myin.open(argv[3]);
            if (!myin)
            {
                cout << "Input matrix File not found \n";
                return 0;
            }
            vector<vf> inmatrix = obj.readmatrix(myin);
            myin.close();
            // opening file in myin , taking input by calling readmatrix

            myin.open(argv[4]);
            if (!myin)
            {
                cout << "Weight matrix File not found \n";
                return 0;
            }
            vector<vf> wtmatrix = obj.readmatrix(myin);
            myin.close();

            myin.open(argv[5]);
            if (!myin)
            {
                cout << "Bias matrix File not found \n";
                return 0;
            }
            vector<vf> biasmatrix = obj.readmatrix(myin);

            vector<vf> answer = MKL_fullyconnected(inmatrix, wtmatrix, biasmatrix);

            myout.open(argv[6]);
            obj.printmatrix(myout, answer);
        }

        else
        {
            cout << "The arguments provided do not match any valid instruction type\n";
        }
    }
    else
    {
        std::cout << "No arguments provided\n";
    }
    cin.tie(nullptr);
    std::cout.tie(nullptr);
    // cout << 9/2 ;

    return 0;
}