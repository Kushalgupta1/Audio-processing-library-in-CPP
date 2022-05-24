COMPILER_FLAGS1 = -fopenmp -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl
COMPILER_FLAGS2 = -lopenblas -lpthread -lgfortran

default: yourcode.out
	
# yourcode.out: task1withpthread.cpp
# 	g++ task1withpthread.cpp -pthread -o yourcode.out

# yourcode_mkl.out: mycode1.cpp mklfullyconnected.h mklfullyconnected.cpp Matrixmath.h Matrixmath.cpp
# 	g++ -L$(MKL_BLAS_PATH)/lib/intel64 -I$(MKL_BLAS_PATH)/include mklfullyconnected.cpp mklfullyconnected.h mycode1.cpp Matrixmath.o -o yourcode_mkl.out $(COMPILER_FLAGS1)
	
yourcode.out : yourcode.cpp libaudio.so
	unset LD_LIBRARY_PATH
	g++ -L . -L$(MKL_BLAS_PATH)/lib/intel64 -Wl,-rpath=. -Wall -o yourcode.out yourcode.cpp -laudio $(COMPILER_FLAGS1)

libaudio.so : audio.o Matrixmath.o mklfullyconnected.o
	g++ -shared -o libaudio.so audio.o Matrixmath.o mklfullyconnected.o

audio.o: audio.h audio.cpp
	g++ -c -Wall -Werror -fpic audio.cpp

Matrixmath.o: Matrixmath.h Matrixmath.cpp
	g++ -c -Wall -Werror -fpic Matrixmath.cpp

mklfullyconnected.o: mklfullyconnected.h mklfullyconnected.cpp 
	g++ -L$(MKL_BLAS_PATH)/lib/intel64 -I$(MKL_BLAS_PATH)/include -c -Wall -Werror -fpic mklfullyconnected.cpp $(COMPILER_FLAGS1)	 

clean:
	rm *.o yourcode.out
