
#ifndef NIVISION_OPENCV_EXAMPLES
#define NIVISION_OPENCV_EXAMPLES

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

EXTERN_C void NI_EXPORT NIVisOpenCV_DetectFaces(NIImageHandle sourceHandle, const char* faceCascadePath, const char* eyesCascadePath, NIArrayHandle facesRectLV, NIArrayHandle eyesRectLV, NIErrorHandle errorHandle);

#endif
