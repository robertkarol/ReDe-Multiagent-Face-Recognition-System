#ifndef OPENCV_WRAPPER
#define OPENCV_WRAPPER

#include <NIVisionExtLib.h>

#include "lv_prolog.h"

typedef struct{
	int kernelSize;
	int scale;
	int delta;	
}LaplacianOptions;

typedef struct {
    int left;
    int top;
    int right;
    int bottom;
    double angle;        
}LV_Rect;

#include "lv_epilog.h"

EXTERN_C void NI_EXPORT NIVisOpenCV_LoadClassifier(const char* faceCascadePath, const char* eyesCascadePath, NIErrorHandle errorHandle);
EXTERN_C void NI_EXPORT NIVisOpenCV_DetectFaces(NIImageHandle sourceHandle, NIArrayHandle facesRectLV, NIArrayHandle eyesRectLV, NIErrorHandle errorHandle);
EXTERN_C void NI_EXPORT NIVisOpenCV_ContainsFace(NIImageHandle sourceHandle, int8_t* containsFace, NIErrorHandle errorHandle);
#endif
