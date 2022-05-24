#include <iostream>
#include "mkl.h"
#include <bits/stdc++.h>
#include <chrono>
using namespace std;
using namespace std::chrono;
typedef long long ll;
typedef vector<vector<float>> vvf;

auto get_time(){
    return std::chrono::high_resolution_clock::now();
}
// This function initialises the 1-D matrix matr in rowmajor order from the vector vec
void init_matrix(vvf &vec , float *matr , int m ,int n){
    for(int i=0 ; i<m ; i++){
        for(int j=0 ; j<n  ; j++){ 
            matr[(i*n) + j] = vec[i][j];
        }
    }
}
// void init_vector()
vvf MKL_fullyconnected(vvf &inmat , vvf&wtmat , vvf& biasmat){
    auto mkl_t1 = get_time();
    // inmat is A*B , wtmat is B*C , biasmat is A*C 
    int A = inmat.size();
    int B= inmat[0].size();
    assert(wtmat.size()==B);
    int C = wtmat[0].size();
    assert(biasmat.size()==A && biasmat[0].size()==C);
  
    float InputMatrix[A*B];
    float WeightMatrix[B*C];
    float BiasMatrix[A*C];

    init_matrix(inmat,InputMatrix , A,B);
    init_matrix(wtmat , WeightMatrix , B ,C);
    init_matrix(biasmat , BiasMatrix , A ,C );

    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,A,C,B,(float)1.0,InputMatrix,B,WeightMatrix,C,(float)1.0,BiasMatrix,C);

    vvf resultmatrix(A , vector<float>(C,0.0));
    for(int i=0 ; i<A ; i++){
        for(int j=0 ; j <C; j++){
            resultmatrix[i][j]=BiasMatrix[C*i + j ]; 
        }
    }
    auto mkl_t2 = get_time();
    // auto mkl_time_span=duration_cast<duration<double>>(mkl_t2-mkl_t1);
    // cout<<"Elapsed time MKL: "<< mkl_time_span.count()<< " s"<< endl;
    // mkl_free(InputMatrix); mkl_free(WeightMatrix); mkl_free(BiasMatrix);
    return resultmatrix ; 

}
