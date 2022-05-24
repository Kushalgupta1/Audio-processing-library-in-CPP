### COP290 Task 1, Subtask 1 

by Kushal Kumar Gupta, 2020CS10355
& Divyanshu Agarwal, 2020CS10343






#### Source Code description and functionalities:



`STL Vectors` with `32 bit float` as datatype to implement matrices. This guarantees a tolerance of`< 1e-5`.



*subtask1withpthread.cpp* provides the following matrix operations:



**1.Fully connected layer:** fully connected (FC) layer that computes inner product of an input matrix of dimensions `AxB` and a weight matrix of dimensions `BxC`, to output a matrix of dimension `AxC`. To this output, a bias vector of dimension `AXC` is then added elementwise. 

Matrix Multiplication types:

- naïve implementation: Stage 1 implementation, simply did the usual multiplication learnt in school O(n^3)

- `pthread`: Multithreading implemented. The number of threads is equal to the number of rows of input matrix (i.e. the dimension of the product matrix). Each thread computes one row of the product matrix.
- `mkl` : Used cblas_sgemm function from Intel's One API Math Kernel library to Multiply and add bias to the matrix.
- `openBLAS` : Used the openblas library, available at openblas.net to perform the Multiplication and adding bias.


**2.Non-linear activations: **Non-linear activations of an input matrix of any size with [`relu`](https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning) and [`tanh`](https://en.wikipedia.org/wiki/Hyperbolic_function) functions on individual matrix elements. tanh has been defined as `tanh(x) = (exp(2x)-1)/(exp(2x)+1)` 



**3.Subsampling with pooling:** Subsampling of square input matrices of any size with [`max pooling`](https://computersciencewiki.org/index.php/Max-pooling_/_Pooling) and `average pooling` functions.  Stride length is required as a parameter for the pooling process. The output matrix is a square matrix of dimension `(n/ Stride)`



**4.Softmax and Sigmoid functions:** Converting a vector of random floats to a vector of probabilities with [`softmax and sigmoid functions`](http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/).



*(Ref: [COP290 webpage](https://www.cse.iitd.ac.in/~rijurekha/cop290_2022.html))*



*mklfullyconnected.cpp* and *blasfullyconnected*  also computes the Fully connected layer as described above but instead use MKL and openblas libraries respectively for Matrix Multiplication.





#### Input/Output data format:



Each input matrix/ vector should be stored in a different file. The output matrix/ vector is stored in the file whose name specified by the user. Note that the output file specified is overwritten if it exists or a new file is created if such a does not exist. The input and output file names/paths are passed as command line arguments.



The following file format is followed for all input/output matrix files: the first line will contain an integer denoting the column dimension, and the second line will denote row dimension. From the third line onwards, single float values are given in column major order in each line of the file. 



Similarly, for vector files: the first line will contain an integer denoting the length, and from the second line onwards, single float values are given in each line of the file. 



*(Ref: [Piazza note](https://piazza.com/class/kyjp5vccd2i7dg?cid=15))*



#### How to use:



1.)For using any of the above functionalities go to the directory of source code (which also contains the `Makefile)` on your terminal and type on the terminal:



```bash

make

```

For compilation of the makefile, we need to ensure that the environment variables 'MKL_INCLUDE_DIR' , 'MKL_LIB_DIR', 'BLAS_INCLUDE_DIR' and 'BLAS_LIB_DIR' contain the respective paths of the include,library of the mkl and openblas implementations respectively.
The environment variable LD_LIBRARY_PATH must also contain the path to the directory containing both the MKL and OpenBLAS library directory. 



This generates executables `yourcode.out, yourcode_mkl.out, yourcode_openblas.out`



Now,



1.)*For using any of the above functionalities **except fullyconnected with mkl or blas**:*



###### 1.Fully connected layer (naive and pthread)



For naïve implementation, type on the terminal



```bash

./yourcode.out fullyconnected inputmatrix.txt weightmatrix.txt biasmatrix.txt outputmatrix.txt

```



For pthread implementation, type on the terminal



```bash

./yourcode.out fullyconnected pthread inputmatrix.txt weightmatrix.txt biasmatrix.txt outputmatrix.txt

```





where `inputmatrix.txt, weightmatrix.txt` and  `biasmatrix.txt` are placeholders for your files names (along with path if not in the current directory) having the input matrix, the weight matrix and the bias matrix respectively, as described above. 



The output matrix is stored in the file name given by the placeholder `outputmatrix.txt`.







###### 2.Non-linear activations



For applying the `relu` function, type on the terminal



```bash

./yourcode.out activation relu inputmatrix.txt outputmatrix.txt

```



For applying the `tanh` function, type on the terminal



```bash

./yourcode.out activation tanh inputmatrix.txt outputmatrix.txt

```



where `inputmatrix.txt` is a placeholder for your files name (along with path if not in the current directory) having the input matrix as described above. 



The output matrix is stored in the file name given by the placeholder `outputmatrix.txt`.







###### 3.Subsampling with pooling



For subsampling of square input matrices of any size with `max pooling`, type on the terminal



```bash

./yourcode.out pooling max inputmatrix.txt stride outputmatrix.txt

```







For subsampling of square input matrices of any size with `average pooling`, type on the terminal



```bash

./yourcode.out pooling average inputmatrix.txt stride outputmatrix.txt

```







where `inputmatrix.txt` is a placeholder for your files name (along with path if not in the current directory), having the input matrix as described above.  `stride` is the positive integer parameter required for the pooling process and should be a factor of input matrix row length and column length.



The output matrix is stored in the file name given by the placeholder `outputmatrix.txt`.







###### 4.Softmax and Sigmoid functions



For converting a vector of random floats to a vector of probabilities with `softmax`, type on the terminal



```bash

./yourcode.out probability softmax inputvector.txt outputvector.txt

```



For converting a vector of random floats to a vector of probabilities with `sigmoid`, type on the terminal



```bash

./yourcode.out probability sigmoid inputvector.txt outputvector.txt

```



where `inputvector.txt` is a placeholder for your files name (along with path if not in the current directory), having the input vector as described above. 



The output vector is stored in the file name given by the placeholder `outputvector.txt`.



2.)*Fullyconnected with MKL:*

Type on the terminal,



```

make

```

This will make an executable yourcode_mkl.out

To run the executable with arguments provided, use the command 

```
./yourcode_mkl.out fullyconnected mkl inputmatrix.txt weightmatrix.txt biasmatrix.txt outputmatrix.txt

```
where `inputmatrix.txt, weightmatrix.txt` and  `biasmatrix.txt` are placeholders for your files names (along with path if not in the current directory) having the input matrix, the weight matrix and the bias matrix respectively, as described above. 
The output matrix is stored in the file name given by the placeholder `outputmatrix.txt`.



3.)*Fullyconnected with OpenBLAS:*

Type on the terminal,



```

make

```

This will make an executable yourcode_openblas.out

To run the executable with arguments provided, use the command 

```
./yourcode_openblas.out fullyconnected openblas inputmatrix.txt weightmatrix.txt biasmatrix.txt outputmatrix.txt

```
where `inputmatrix.txt, weightmatrix.txt` and  `biasmatrix.txt` are placeholders for your files names (along with path if not in the current directory) having the input matrix, the weight matrix and the bias matrix respectively, as described above. 
The output matrix is stored in the file name given by the placeholder `outputmatrix.txt`.
