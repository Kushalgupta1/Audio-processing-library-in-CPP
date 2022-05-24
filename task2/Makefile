COMPILER_FLAGS1 = -fopenmp -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
COMPILER_FLAGS2 = -lopenblas -lpthread -lgfortran

default: yourcode.out yourcode_mkl.out yourcode_openblas.out
	
yourcode.out: task1withpthread.cpp
	g++ task1withpthread.cpp -pthread -o yourcode.out

yourcode_mkl.out: mycode1.cpp mklfullyconnected.h mklfullyconnected.cpp Matrixmath.h Matrixmath.cpp
	g++ -L$(MKL_LIB_DIR) -I$(MKL_INCLUDE_DIR) mklfullyconnected.cpp mklfullyconnected.h mycode1.cpp Matrixmath.o -o yourcode_mkl.out $(COMPILER_FLAGS1)
	
yourcode_openblas.out: mycode2.cpp blasfullyconnected.h blasfullyconnected.cpp Matrixmath.h Matrixmath.cpp
	g++ -o yourcode_openblas.out blasfullyconnected.cpp blasfullyconnected.h mycode2.cpp Matrixmath.o -I$(BLAS_INCLUDE_DIR) -L$(BLAS_LIB_DIR) $(COMPILER_FLAGS2)

Matrixmath.o: Matrixmath.h Matrixmath.cpp
	g++ -c Matrixmath.cpp

clean:
	rm *.o yourcode.out
