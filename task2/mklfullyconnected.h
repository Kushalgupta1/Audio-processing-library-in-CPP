#ifndef MKLFULL_H
#define MKLULL_H
using namespace std;
#include <iostream>
#include "mkl.h"
#include <bits/stdc++.h>
#include <chrono>
using namespace std;
using namespace std::chrono; 
typedef long long ll;
typedef vector<vector<float>> vvf;



auto get_time();
void init_matrix(vvf &vec , float *matr , int m ,int n);
vvf MKL_fullyconnected(vvf &inmat , vvf&wtmat , vvf& biasmat);
  
#endif 