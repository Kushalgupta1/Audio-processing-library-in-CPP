#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>
#include <vector>
#include <algorithm>
#include <functional>
#include <numeric>
#include <exception> 
#include <cmath>
#include <pthread.h>

int row_curr = 0;

//function to convert from char array to int
bool str2int (char *str, int &i){
    char t;
    std::stringstream ss(str);
    ss >> i;
    if (ss.fail() || ss.get(t)) {
        //characters left in input str, hence not an integer
        return false;
    }
    return true;
}

//templated tanh, relu, sigmoid
template<typename T> 
T myTanh( T x){
	T ExpOf2x = exp(2*x);
	return (ExpOf2x - 1)/(ExpOf2x + 1);
}

template<typename T> 
T myRelu(T x){
	if(x <= 0) return 0;
	else return x;
}

template<typename T> 
T mySigmoid(T x){
	T negExp = exp(-x);
	return 1/(1 + negExp);
}

template<typename T> 
T myExp(T x){
	return exp(x);
}

//templated class for matrix
//create a matrix object, eg, MatrixUsingVectors<float> MAT(5, 10)
template<typename T> 
class MatrixUsingVectors{
	//store matrix as an std::vector 
	std::vector <T> arrOfMatrix;

public:
	int rows, cols;
	MatrixUsingVectors(int row_count, int col_count){
		rows = row_count;
		cols = col_count;
		arrOfMatrix.resize(rows * cols);
	}
	//operator() returns reference to matrix element at index, eg, if MAT is the matrix, access element at 2,3 index using MAT(2,3)
    T& operator()(int row_no, int col_no)
    {
        return arrOfMatrix[(row_no * cols) + col_no];
    }	
	//inner product of two compatible matrices
	void multiplyBy(MatrixUsingVectors &B, MatrixUsingVectors &res){
		for (int rowA = 0; rowA < this->rows; rowA++){
			for (int colB = 0; colB < B.cols; colB++){
				//compute product[rowA][rowB]
				T sum = 0;
				for (int i = 0; i < this->cols; i++){
					sum+= (*this)(rowA, i) * B(i, colB);
				}
				res(rowA, colB) = sum;
			}
		}
	}


	//add 2 matrices
	void add(MatrixUsingVectors &B, MatrixUsingVectors &res){
		std::transform(this->arrOfMatrix.begin(),  this->arrOfMatrix.end(), B.arrOfMatrix.begin(), res.arrOfMatrix.begin(), std::plus<T>() );
	}

	//apply any function to the matrix element-wise (used to apply tanh, relu, sigmoid)
	void applyFunction(MatrixUsingVectors &res, std::function< T (T)> func){ 
		std::transform(this->arrOfMatrix.begin(),this->arrOfMatrix.end(), res.arrOfMatrix.begin(), func);
	}	

	//apply softmax
	void applySoftmax(MatrixUsingVectors &res){
		applyFunction(res, myExp<T>);
		T sum = 0;
		sum = std::accumulate(res.arrOfMatrix.begin(), res.arrOfMatrix.end(), sum);
		for(auto &x: res.arrOfMatrix) 
			x = x/sum;
	}

	//Pooling Max or Pooling average depending on toggle
	void PoolingMaxAvgBar(MatrixUsingVectors &res, int stride, bool toggle){
		//toggle = true => Pool max, toggle = false => pool average
		//res(i, j) will contain max.average of entries from rows stride*i to stride*(i+1) - 1 and cols stride*j to stride*(j +1) - 1
		for (int i = 0; i < res.rows; i++)
		{
			for (int j = 0; j < res.cols; j++)
			{
				T maxOrSum;
				if(toggle) maxOrSum = (*this)(stride * i, stride * j);
				else maxOrSum = 0;

				for (int rowA = stride*i; rowA < stride*(i+1); rowA++)
				{
					for (int colA = stride * j; colA < stride*(j +1); colA++)
					{
						if(toggle) maxOrSum = ( ((*this)(rowA, colA) > maxOrSum) ? (*this)(rowA, colA) : maxOrSum);
						
						else maxOrSum+= (*this)(rowA, colA);
					}
				}
				if(toggle) res(i, j) = maxOrSum;
				else res(i, j) = maxOrSum / (stride * stride);
			}
		}
	}

};

//multiply the global matrices A_pthread and B_pthread and store the result in res_pthread
MatrixUsingVectors<float> *A_pthread; //Matrices for fullyconnected with pthread
MatrixUsingVectors<float> *B_pthread; //Matrices for fullyconnected with pthread
MatrixUsingVectors<float> *res_pthread; 
void* multiplyARow(void *noUse);

void multiplyBypthread(){

pthread_t threads[A_pthread->rows];

// Creating total A_pthread row number of threads, each evaluating that row in the product
for (int i = 0; i < A_pthread->rows; i++) {
	int* p;
	pthread_create(&threads[i], NULL, multiplyARow, (void*)(p));
}

// joining and waiting for all threads to complete
for (int i = 0; i < A_pthread ->rows; i++)
	pthread_join(threads[i], NULL);	

}

void* multiplyARow(void *noUse){
	// computes all elements in the row row_curr of the result 
	int rowA = row_curr++;

	for (int colB = 0; colB < B_pthread->cols; colB++)
	{
		// compute product[rowA][rowB]
		float sum = 0;
		for (int i = 0; i < A_pthread->cols; i++){
			sum += (*A_pthread)(rowA, i) * (*B_pthread)(i, colB);
			//std:: cout << "sum after"
		}
			(*res_pthread)(rowA, colB) = sum;
		}
	return NULL;
}



int main(int argc, char *argv[]){

	try{
		if (argc < 2){
			throw 401;
		}
		else{
			//FULLY CONNECTED, normal multiplication and multithreaded---------------------------------------
			int cmp = strcmp(argv[1], "fullyconnected");
			if (cmp == 0) {
				if(argc == 6){ 
					std::ifstream InpMatFile(argv[2]);
					std::ifstream WeightMatFile(argv[3]);
					std::ifstream BiasMatFile(argv[4]);

					if(InpMatFile.good() && WeightMatFile.good() && BiasMatFile.good()){
						std::ofstream OutMatFile(argv[5]);
						//implementing FC operation
						int rows, cols;
						InpMatFile >> cols >> rows;
						MatrixUsingVectors<float> inpMAT(rows, cols);
						WeightMatFile >> cols >> rows;
						MatrixUsingVectors<float> weightMAT(rows, cols);
						BiasMatFile >> cols >> rows;
						MatrixUsingVectors<float> biasMAT(rows, cols);
						std::cout << cols << "\n" << rows << "\n";
						MatrixUsingVectors<float> outMAT(biasMAT.rows, biasMAT.cols);

						for (int j = 0; j < inpMAT.cols; j++)
							for (int i = 0; i < inpMAT.rows; i++){
								InpMatFile >> inpMAT(i, j);
							}

						for (int j = 0; j < weightMAT.cols; j++)
							for (int i = 0; i < weightMAT.rows; i++)
								WeightMatFile >> weightMAT(i, j);

						for (int j = 0; j < biasMAT.cols; j++)
							for (int i = 0; i < biasMAT.rows; i++)
								BiasMatFile >> biasMAT(i, j);

						inpMAT.multiplyBy(weightMAT, outMAT);
						outMAT.add(biasMAT, outMAT);
						OutMatFile << outMAT.cols << "\n" << outMAT.rows << "\n";
						std::cout << outMAT.cols << "\n" << outMAT.rows << "\n";

						for (int j = 0; j < outMAT.cols; j++)
							for (int i = 0; i < outMAT.rows; i++){
								OutMatFile << outMAT(i, j) << "\n";	
							}

						return 1;
					}
					//file not accessible
					else throw 201;
				}

				else if(argc == 7){ 
					int cmp1 = strcmp(argv[2], "pthread");
					if(cmp1 != 0) throw 411; //incorrect fully connected parameter

						std::ifstream InpMatFile(argv[3]);
						std::ifstream WeightMatFile(argv[4]);
						std::ifstream BiasMatFile(argv[5]);

						if(InpMatFile.good() && WeightMatFile.good() && BiasMatFile.good()){
							std::ofstream OutMatFile(argv[6]);
							//implementing FC operation

							int rows, cols;
							InpMatFile >> cols >> rows;
							A_pthread = new MatrixUsingVectors<float>(rows, cols); //inpMAT
							WeightMatFile >> cols >> rows;
							B_pthread = new MatrixUsingVectors<float>(rows, cols); //weightMAT
							BiasMatFile >> cols >> rows;
							MatrixUsingVectors<float> biasMAT(rows, cols); 
							res_pthread = new MatrixUsingVectors<float>(rows, cols);

							for (int j = 0; j < A_pthread-> cols; j++)
								for (int i = 0; i < A_pthread-> rows; i++)
									InpMatFile >> (*A_pthread)(i, j);

							for (int j = 0; j < B_pthread -> cols; j++)
								for (int i = 0; i < B_pthread -> rows; i++)
									WeightMatFile >> (*B_pthread)(i, j);

							for (int j = 0; j < biasMAT.cols; j++)
								for (int i = 0; i < biasMAT.rows; i++)
									BiasMatFile >> biasMAT(i, j);


							row_curr = 0;
							multiplyBypthread();
							(*res_pthread).add(biasMAT, *res_pthread);
							OutMatFile << res_pthread->cols << "\n" << res_pthread->rows << "\n";

							for (int j = 0; j < res_pthread->cols; j++)
								for (int i = 0; i < res_pthread->rows; i++){
									OutMatFile << (*res_pthread)(i, j) << "\n";	
								}

							return 1;
						}
						//file not accessible
						else throw 201;
				}
				else throw 401;
			}
			
			//ACTIVATION--------------------------------------
			cmp = strcmp(argv[1], "activation");
			if (cmp == 0) {
				if(argc != 5) throw 401;
				else {
					auto func = myTanh<float>;
					int cmp1 = strcmp(argv[2], "tanh");
					int cmp2 = strcmp(argv[2], "relu");
					if(cmp1 == 0) ;
					else if(cmp2 == 0) func = myRelu<float>;
					else throw 403; //invalid activation parameter

					std::ifstream InpMatFile(argv[3]);

					if (InpMatFile.good())
					{
						std::ofstream OutMatFile(argv[4]);
						//implementing tanh or relu function
						int rows, cols;
						InpMatFile >> cols >> rows;
						MatrixUsingVectors<float> inpMAT(rows, cols);

						for (int j = 0; j < inpMAT.cols; j++)
							for (int i = 0; i < inpMAT.rows; i++)
								InpMatFile >> inpMAT(i, j);

						inpMAT.applyFunction(inpMAT, func);
						OutMatFile << inpMAT.cols << "\n" << inpMAT.rows << "\n";

						for (int j = 0; j < inpMAT.cols; j++)
							for (int i = 0; i < inpMAT.rows; i++)
								OutMatFile << inpMAT(i, j) << "\n";	
						
						return 1;
					}
					else throw 201; //file not accessible
				}
			}
			//POOLING-----------------------------------------------------------
			cmp = strcmp(argv[1], "pooling");
			if (cmp == 0) {
				if(argc != 6) throw 401;
				else{
					bool toggle = true; //toggle for whether to compute max or average
					int cmp1 = strcmp(argv[2], "max");
					int cmp2 = strcmp(argv[2], "average");
					if(cmp1 == 0) ;
					else if(cmp2 == 0) toggle = false;
					else throw 404; //invalid pooling parameter

					std::ifstream InpMatFile(argv[3]);

					if (InpMatFile.good())
					{
						int stride;
						if(!str2int(argv[4], stride)) throw 405;
						std::ofstream OutMatFile(argv[5]);

						int rows, cols;
						InpMatFile >> cols >> rows;
						MatrixUsingVectors<float> inpMAT(rows, cols);

						if(stride == 0 || !( (rows % stride == 0) && (cols % stride == 0) )) throw 405;
						MatrixUsingVectors<float> outMAT((int) (rows/stride), (int) (cols/stride));

						for (int j = 0; j < inpMAT.cols; j++)
							for (int i = 0; i < inpMAT.rows; i++)
								InpMatFile >> inpMAT(i, j);

						inpMAT.PoolingMaxAvgBar(outMAT, stride, toggle);
						OutMatFile << outMAT.cols << "\n" << outMAT.rows << "\n";

						for (int j = 0; j < outMAT.cols; j++)
							for (int i = 0; i < outMAT.rows; i++)
								OutMatFile << outMAT(i, j) << "\n";	

						return 1;
					}
					else throw 201; //file not accessible
				}
				
			}
			//PROBABILITY-----------------------------------------------------------
			cmp = strcmp(argv[1], "probability");
			if (cmp == 0) {
				if(argc != 5) throw 401;
				else {
					bool toggle = true; //true for softmax, false for sigmoid
					int cmp1 = strcmp(argv[2], "softmax");
					int cmp2 = strcmp(argv[2], "sigmoid");
					if(cmp1 == 0) ;
					else if(cmp2 == 0) toggle = false;
					else throw 406; //invalid probability parameter

					std::ifstream InpMatFile(argv[3]);

					if (InpMatFile.good())
					{
						std::ofstream OutMatFile(argv[4]);
						//implementing softmax or sigmoid function
						int cols;
						InpMatFile >> cols;
						MatrixUsingVectors<float> inpMAT(1, cols);

						for (int j = 0; j < inpMAT.cols; j++)
							for (int i = 0; i < inpMAT.rows; i++)
								InpMatFile >> inpMAT(i, j);

						if(toggle) {
							//softmax
							inpMAT.applySoftmax(inpMAT);
						}
						else{
							inpMAT.applyFunction(inpMAT, mySigmoid<float>);
						}
						OutMatFile << inpMAT.cols << "\n";

						for (int j = 0; j < inpMAT.cols; j++)
							for (int i = 0; i < inpMAT.rows; i++)
								OutMatFile << inpMAT(i, j) << "\n";	
						
						return 1;
					}
					else throw 201; //file not accessible
				}
			}
			//2nd parameter not matching------------------------------------------
			throw 402;
		}
	}
	catch(int err_code){
		if(err_code == 401){
			std::cerr << "Error: incorrect number of arguments, refer to readme for help";
		}
		if(err_code == 402){
			std::cerr << "Error: invalid control parameter(should be one of fullyconnected, activation, pooling, probability), refer to readme for help";
		}
		if(err_code == 411){
			std::cerr << "Error: invalid control parameter for fullyconnected(there should be no parameter or pthread), refer to readme for help";
		}
		if(err_code == 403){
			std::cerr << "Error: invalid control parameter for activation(should be one of tanh or relu), refer to readme for help";
		}
		if(err_code == 404){
			std::cerr << "Error: invalid control parameter for pooling(should be one of max or average), refer to readme for help";
		}
		if(err_code == 405){
			std::cerr << "Error: invalid value of stride, should be a positive integer and a factor of input matrix rowlength and column length, refer to readme for help";
		}
		if(err_code == 406){
			std::cerr << "Error: invalid control parameter for probability(should be one of softmax or sigmoid), refer to readme for help";
		}
		if(err_code == 201){
			std::cerr << "Error: unable to access input file(s), check file path or file access permissions or refer to readme for help";
		}
	}
}
