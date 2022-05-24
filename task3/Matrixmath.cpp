#include <vector>
#include "Matrixmath.h"
using namespace std;
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <math.h>
typedef vector<float> vf;

float Matrixmath::mytanh(float a)
{
    float c = exp(2 * a);
    float d = (c - 1) / (c + 1);
    return d;
}
float Matrixmath::mysigmoid(float a)
{
    float b = exp(a);
    return (b / (b + 1));
}

vector<vf> Matrixmath::fullyconnected(vector<vf> &inmat, vector<vf> &wtmat, vector<vf> &biasmat)
{
    int A = inmat.size();
    int B = inmat[0].size();
    // if(wtmat.size()!=B){cout<<"Wrong dimensions of matrices to be multiplied\n";return {};}
    // assert(wtmat.size() == B);
    int C = wtmat[0].size();

    // try{if(biasmat.size()!=A || biasmat[0].size()!=C) {throw A;}}catch(int A){cout<<"Wrong dimensions found while adding bias and product matrix\n";return {};}
    // assert((biasmat.size() == A) && (biasmat[0].size() == C));
    // Mutliplication  of matrices
    vector<vector<float>> outmat(A, vector<float>(C, 0));
    for (int i = 0; i < A; i++)
    {
        for (int j = 0; j < C; j++)
        {
            float sum = 0;
            for (int k = 0; k < B; k++)
            {
                sum += inmat[i][k] * wtmat[k][j];
            }
            outmat[i][j] = sum;
        }
    }

    // Adding bias to the multiplied product of the matrices
    // assert(outmat.size() == A && outmat[0].size() == C);
    for (int i = 0; i < A; i++)
    {
        for (int j = 0; j < C; j++)
        {
            outmat[i][j] += biasmat[i][j];
        }
    }
    return outmat;
}
vector<vf> Matrixmath::activation_relu(vector<vf> inmat)
{
    int A = inmat.size();
    int B = inmat[0].size();

    for (int i = 0; i < A; i++)
    {
        for (int j = 0; j < B; j++)
        {
            inmat[i][j] = max(0.0f, inmat[i][j]);
        }
    }
    // assert(inmat.size() == A && inmat[0].size() == B);
    return inmat;
}
vector<vf> Matrixmath::activation_tanh(vector<vf> inmat)
{
    int A = inmat.size();
    int B = inmat[0].size();
    for (int i = 0; i < A; i++)
    {
        for (int j = 0; j < B; j++)
        {
            inmat[i][j] = mytanh(inmat[i][j]);
        }
    }
    // assert(inmat.size() == A && inmat[0].size() == B);
    return inmat;
}
vector<vf> Matrixmath::max_pool(vector<vf> &inmat, int stride)
{
    int A = inmat.size();

    // assert(inmat[0].size() == A && (A % stride == 0));

    // Checking square matrix condition and that matrix size is a multiple of stride value

    int D = A / stride;
    // dimension of new matrix

    vector<vector<float>> outmat(D, vector<float>(D));
    for (int i = 0; i < D; i++)
    {
        for (int j = 0; j < D; j++)
        {
            int m = (stride * i);
            int n = (stride * j);
            // m and n contain the start point of the region whose max has to be calculated
            float curr_max = inmat[m][n];
            // initialise the currentmaximum as the value of the first element of the region , we subsequently update this maximum by comparing with all elements in the square region.

            for (int p = m; p < m + stride; p++)
            {
                for (int q = n; q < n + stride; q++)
                {
                    curr_max = max(curr_max, inmat[p][q]);
                }
            }
            outmat[i][j] = curr_max;
            // assigning value to elements of outmat as the maximum value over the region .
        }
    }
    // assert((outmat.size() == outmat[0].size()) && (outmat.size() == A / stride));
    return outmat;
}
vector<vf> Matrixmath::average_pool(vector<vf> &inmat, int stride)
{
    int A = inmat.size();
    // assert(inmat[0].size() == A && (A % stride == 0));
    // Checking square matrix condition and that matrix size is a multiple of stride value

    int D = A / stride;
    // dimension of new matrix

    vector<vector<float>> outmat(D, vector<float>(D));
    for (int i = 0; i < D; i++)
    {
        for (int j = 0; j < D; j++)
        {
            int m = (stride * i);
            int n = (stride * j);
            // m and n contain the start point of the region whose max has to be calculated
            float sum = 0;
            // initialise the currentmaximum as the value of the first element of the region , we subsequently update this maximum by comparing with all elements in the square region.

            for (int p = m; p < m + stride; p++)
            {
                for (int q = n; q < n + stride; q++)
                {
                    sum += inmat[p][q];
                }
            }

            outmat[i][j] = sum / ((float)(stride * stride));
            // assigning value to elements of outmat as the maximum value over the region .
        }
    }
    // assert((outmat.size() == outmat[0].size()) && (outmat.size() == A / stride));
    return outmat;
}
vf Matrixmath::probability_sigmoid(vf invec)
{
    for (long unsigned int i = 0; i < invec.size(); i++)
    {
        invec[i] = mysigmoid(invec[i]);
    }
    return invec;
}
vf Matrixmath::probability_softmax(vf invec)
{
    float sum = 0;
    for (long unsigned int i = 0; i < invec.size(); i++)
    {
        float a = exp(invec[i]);
        sum += a;
        invec[i] = a;
    }
    for (long unsigned int i = 0; i < invec.size(); i++)
    {
        invec[i] /= sum;
    }
    return invec;
}

vector<vf> Matrixmath::readmatrix(ifstream &myin)
{
    // assert(myin);
    int n, m;
    myin >> m >> n;
    vector<vf> matrix(n, vf(m, 0.0));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            myin >> matrix[j][i];
        }
    }
    return matrix;
}
void Matrixmath::printmatrix(ofstream &myout, const vector<vf> &matrix)
{
    // assert(myout);
    int m = matrix[0].size();
    int n = matrix.size();
    myout << m << "\n";
    myout << n << "\n";

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            myout << matrix[j][i] << "\n";
        }
    }
}
vf Matrixmath::readvector(ifstream &myin)
{
    // assert(myin);
    int n;
    myin >> n;
    vf myvector(n, 0.0);
    for (int i = 0; i < n; i++)
    {
        myin >> myvector[i];
    }
    return myvector;
}
void Matrixmath::printvector(ofstream &myout, const vf &vector)
{
    // assert(myout);
    int n = vector.size();
    myout << n << "\n";

    for (int i = 0; i < n; i++)
    {
        myout << vector[i] << "\n";
    }
}

vector<vf> Matrixmath::RowMajorToMatrix(float arr[] , int n , int m){
    vector<vf> ans(n,vf(m,0.0));
    for(int i=0 ; i<n;i++){
        for(int j=0 ; j<m;j++){
            ans[i][j]=arr[i*m+j];
        }
    }
    return ans;
}

 void Matrixmath::RowMajorFromFile(ifstream &myin,int n , int m ,vector<vf>& location){
     for(int i=0 ; i<n;i++){
         for(int j=0;j<m;j++){
             myin>>location[i][j];
         }
     }
 };
