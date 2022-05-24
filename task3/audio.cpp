#include "dnn.h"
#include "audio.h"
#include "Matrixmath.h"
#include "mklfullyconnected.h"
#include <fstream>

using namespace std;
#include<bits/stdc++.h>

float wt1Arr[] = IP1_WT;
float wt2Arr[] = IP2_WT;
float wt3Arr[] = IP3_WT;
float wt4Arr[] = IP4_WT;

float bias1Arr[] = IP1_BIAS;
float bias2Arr[] = IP2_BIAS;
float bias3Arr[] = IP3_BIAS;
float bias4Arr[] = IP4_BIAS;


Matrixmath obj ;
ifstream myin;

using namespace std;  
  
// float featureArray[250]; // define max string  // as of now I have assumed ki 250 size mein store karna hai
  
// // length of the string  
// int len(string str)  
// {  
//     int length = 0;  
//     for (int i = 0; str[i] != '\0'; i++)  
//     {  
//         length++;  
          
//     }  
//     return length;     
// }  
  
// create custom split() function  
// void split (string str, char seperator)  
// {  
//     int currIndex = 0, i = 0;  
//     int startIndex = 0, endIndex = 0;  
//     while (i <= len(str))  
//     {  
//         if (str[i] == seperator || i == len(str))  
//         {  
//             endIndex = i;  
//             string subStr = "";  
//             subStr.append(str, startIndex, endIndex - startIndex);  
//             featureArray[currIndex] = stof(subStr);  
//             currIndex += 1;  
//             startIndex = endIndex + 1;  
//         }  
//         i++;  
//         }     
// }  


    pred_t::pred_t(){    
        label=0;prob=0;
    }

    pred_t::pred_t(int x , float y){
        label=x ; prob=y;
    }

pred_t* libaudioAPI(const char* audiofeatures, pred_t* pred){
        myin.open(audiofeatures);
        vector<vf> initial(1 ,vf(250,0.0));
        obj.RowMajorFromFile(myin,1,250,initial);
        myin.close();

        vector<vf>wtmat1 = obj.RowMajorToMatrix( wt1Arr,250,144);
        vector<vf>wtmat2 = obj.RowMajorToMatrix(wt2Arr,144,144);
        vector<vf>wtmat3 = obj.RowMajorToMatrix(wt3Arr,144,144);
        vector<vf>wtmat4 = obj.RowMajorToMatrix(wt4Arr,144,12);

        vector<vf>biasmat1 = obj.RowMajorToMatrix(bias1Arr,1,144);
        vector<vf>biasmat2 = obj.RowMajorToMatrix(bias2Arr,1,144);
        vector<vf>biasmat3 = obj.RowMajorToMatrix(bias3Arr,1,144);
        vector<vf>biasmat4 = obj.RowMajorToMatrix(bias4Arr,1,12);

        // vector<vf>initial = obj.RowMajorToMatrix(featureArray,1,250);

        vector<vf> result1 = obj.activation_relu(MKL_fullyconnected(initial , wtmat1 , biasmat1));
        vector<vf> result2 = obj.activation_relu(MKL_fullyconnected(result1 , wtmat2 , biasmat2));
        vector<vf> result3 = obj.activation_relu(MKL_fullyconnected(result2 , wtmat3 , biasmat3));
        vf result4 = obj.probability_softmax(MKL_fullyconnected(result3 , wtmat4 , biasmat4)[0]);

        vf probabilities = result4;
        map<float , int> m ; 
        for(int i=0 ; i<12 ; i++){
            m[probabilities[i]]=i;
        }


        auto it = m.end() ; 
        it--;
        pred[0]=pred_t(it->second,it->first);
        it--;
        pred[1]=pred_t(it->second,it->first);
        it--;
        pred[2]=pred_t(it->second,it->first);

    return pred;
    // Just use the best implementaion and do the matrix multiplication , then apply the relu funciton on the vector you obtain after multiplying.
    // Do this for the 3 layers and we are done, just return the top 3 probablities with 
        
// inputmatrix will be 1x250 , 250x144 bias se multiply karna hai , bias addkarna hai? , then relu , then 144x144 wala multiplication 2 baar aur fir 144*12 wala . uske 1*12 ka vector milega uski top 3 probabilites ko return karna hai
};

// We have to mention #include "audio.h" in teh file where we want to use this 
//compile karte time we have to do $ g++ -c -Wall -Werror -fpic audio.cpp
// this is dohttps://www.cse.iitd.ac.in/~rijurekha/cop290_2022.htmlne to compile the library, then 
//$ g++ -shared -o libaudio.so audio.o  
//this is done to turn the object library into shared library 
// when we are compiling out code do 
// $ g++ -L<path of the directory containing library> -Wall -o test main.cpp -laudio
// -L tells the linker where to find the library and -laudio tells to look for libaudio.so

// finally when the executable has to be run the loader need to know the lobrary location for this do 
// $ export LD_LIBRARY_PATH= <path of library dierctory>:$LD_LIBRARY_PATH
// then do $ ./test     ---> to execute your file.

// int main(){
//     pred_t  inp[3];
//     pred_t* answer = libaudioAPI("3c257192_nohash_3.txt",inp);
    
    
//     cout<<"Label "<<answer[0].label<<" has probability"<<" "<<answer[0].prob<<endl;
//     cout<<"Label "<<answer[1].label<<" has probability"<<" "<<answer[1].prob<<endl;
//     cout<<"Label "<<answer[2].label<<" has probability"<<" "<<answer[2].prob<<endl;

// }