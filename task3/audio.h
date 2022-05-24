#ifndef audio_h__ 
#define audio_h__
struct pred_t{
int label;
float prob;

pred_t();
pred_t(int x , float y);

};

extern pred_t* libaudioAPI(const char* audiofeatures, pred_t* pred);

#endif