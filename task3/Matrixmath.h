#ifndef MATRIXMATH_H 
#define MATRIXMATH_H
using namespace std;
#include<vector>
#include <string>
#include<iostream>
#include<fstream>
#include<cassert>
#include<math.h>
typedef vector<float> vf;
class Matrixmath{ 
    vector<vf> mymartix;
    vf myvector;

    float mytanh(float a );
    float mysigmoid(float a);
    
    public :
        
        vector<vf> fullyconnected(vector<vf> &inmat , vector<vf> &wtmat , vector<vf> &biasmat );
        vector<vf> activation_relu(vector<vf> inmat);
        vector<vf> activation_tanh(vector<vf> inmat);
        vector<vf> max_pool(vector<vf> &inmat , int stride);
        vector<vf> average_pool(vector<vf> &inmat , int stride);
        vf probability_sigmoid(vf invec);
        vf probability_softmax(vf invec);


        vector<vf> readmatrix(ifstream &myin);
        void printmatrix(ofstream &myout , const vector<vf> &matrix);
        vf readvector(ifstream &myin);
        void printvector(ofstream &myout , const vf &vector);

        vector<vf> RowMajorToMatrix(float arr[] , int n , int m);
        void RowMajorFromFile(ifstream &myin,int n , int m ,vector<vf> &location);
        // Description of the various function here
    
};
#endif 