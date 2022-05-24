### COP290 Task 1, Subtask 3

by Kushal Kumar Gupta, 2020CS10355
& Divyanshu Agarwal, 2020CS10343



Implementing a deep neural network (DNN) inference for classifying across 12 audio keywords (silence, unknown, yes, no, up, down, left, right, on, off, stop, go). 

- A usual ML classification pipeline consists of feature extraction followed by a (fully connected) classifier. We have provided the tensorflow based python script compute_mfcc.py to extract features from the given audio samples. To ease the feature extraction process, you are not required to setup a python environment and extract the features. We also provide you the extracted feature in `mfcc_features.zip`, which you can directly use in further computations.

- The weights (and bias) for various FC components are given in `dnn_weights.h`, in row major order.

- The SoftMax output for various samples is given in `softmax_output.csv`. You can refer this to validate your results.
#### Source Code description and functionalities:

Stitched together Fullyconnected (using MKL Library for matrix multiplication) from subtask 2 and Relu and SoftMax from subtask 1 to implement a DNN which takes a [1x250] input features vector for a 1 second audio sample from the specified input file and applies-

`FC1 [250x144] -> RELU -> FC2 [144x144] -> RELU -> FC3 [144X144] -> RELU -> FC4 [144x12] -> softmax`

The softmax probabilites are obtained for each keyword and the top 3 keywords with highest softmax probabilities, and their softmax probabilities in the same order are reported in the specified output file.



Implementation details of FC, Relu and Softmax can be seen from the current or previous stages' source files.



#### Input/Output data format:

Input file name (with full path and extension) is passed as a command line argument. If the file does not exist, the program exits displaying `Input matrix File not found`. 

This input file should contain 250 floats, which represent the [1x250] input features vector obtained from the feature extraction of a 1 second audio sample. 

Output file name (with full path and extension) is passed as a command line argument. If the file does not exist, the program creates the file.  If output file exists, the output is appended to the end of the file.  The output is-

Input file name (with full path and extension), followed by the top 3 keywords with highest softmax probabilities, and then followed by their softmax probabilities in the same order.

*(Ref: [Piazza note](https://piazza.com/class/kyjp5vccd2i7dg?cid=46))*



#### How to use:

1.)For using the DNN, go to the directory of source code (which also contains the `Makefile)` on your terminal and type on the terminal:

```bash
make
```

**Note:** For compilation of the `makefile`, we need to ensure that the environment variable `MKL_BLAS_PATH` contains the longest common path of the include and library of the mkl implementation.



This generates the executables `yourcode.out`. Now, type on the terminal

```bash
./yourcode.out audiosamplefile outputfile
```



where `audiosamplefile`  is the placeholder for the input files name (along with path if not in the current directory) having the input features vector, as described above. 

The output matrix is stored in the file name given by the placeholder `outputmatrix.txt`. The output is as described above.

**Notes on implementation**:

- `libaudio.so` contains 3 object files `audio.o, Matrixmath.o` and `mklfullyconnected.o` 
- `audio.cpp`  takes weights and bias matrices for FC activation from `dnn.h`
- Defined `RowMajorToMatrix`  and `RowMajorFromFile` to read the matrices from `dnn.h` and given input file

Output file `out.txt` has been included with outputs for some of the input files.

