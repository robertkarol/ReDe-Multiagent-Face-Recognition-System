#ifdef USE_PRECOMPILED_HEADER
#include "stdafx.h"
#endif

#include <vector>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <NIVisionExtLib.h>
#include "FaceDetection.h"
using namespace std;
using namespace cv;

typedef cv::Point2f PointFloat;
typedef cv::Point2d PointDouble;

static CascadeClassifier faceCascadeClassifier;
static CascadeClassifier eyesCascadeClassfier;
static bool bEyeClassifierLoaded = false;

void Convert(const Rect& rect, LV_Rect& lvRect){
    lvRect.left = rect.x;
    lvRect.top = rect.y;
    lvRect.right = rect.x + rect.width - 1;
    lvRect.bottom = rect.y + rect.height - 1;
    lvRect.angle = 0;
}

EXTERN_C void NI_EXPORT NIVisOpenCV_LoadClassifier(const char * faceCascadePath, const char * eyesCascadePath, NIErrorHandle errorHandle)
{
	String faceCascadeString = faceCascadePath;
	if (!faceCascadeClassifier.load(faceCascadeString))
	{
		return;
	}
	String eyesCascadeString = eyesCascadePath;
	if (eyesCascadePath != NULL && eyesCascadeString.length() > 1)
	{
		if (!eyesCascadeClassfier.load(eyesCascadeString))
		{
			return;
		}
	}
}

EXTERN_C void NI_EXPORT NIVisOpenCV_DetectFaces(NIImageHandle sourceHandle, NIArrayHandle facesRectLV, NIArrayHandle eyesRectLV, NIErrorHandle errorHandle){
    
    NIERROR error = NI_ERR_SUCCESS;
    ReturnOnPreviousError(errorHandle);
	
    try
	{
        NIImage source(sourceHandle);        
        NIArray1D<LV_Rect> facesRect(facesRectLV);  

		//Do image conversions
        vector<Rect> faces;
        vector<LV_Rect> facesLV;
        Mat sourceMat;
        Mat faceImage;
        ThrowNIError(source.ImageToMat(sourceMat));
        Mat matGray;
		int maxWidth = matGray.rows / 2;
		int minWidth = 15;
		if (sourceMat.type() != CV_8UC1) 
		{
			cvtColor(sourceMat, matGray, COLOR_BGR2GRAY);
		}
		else 
		{
			matGray = sourceMat.clone();
		}

		//Equalize image
        equalizeHist(matGray, matGray);

		//Detect faces
        faceCascadeClassifier.detectMultiScale(matGray, faces, 1.15, 4, 0 | CASCADE_SCALE_IMAGE, Size(minWidth, minWidth));
        
		facesLV.resize(faces.size());
        vector<LV_Rect>::iterator fLV = facesLV.begin();
		if (faces.size()) 
		{
			for (vector<Rect>::iterator f = faces.begin(); f != faces.end(); f++) 
			{
				Convert(*f, *fLV++);
				if (minWidth > f->width) 
				{
					minWidth = f->width;
				}
				if (maxWidth < f->width) 
				{
					maxWidth = f->width;
				}
			}

			minWidth = static_cast<int>(minWidth * 0.8);
			maxWidth = static_cast<int>(maxWidth * 1.2);

		}
		else 
		{
			minWidth = 15;
			maxWidth = matGray.rows / 4;
		}
        facesRect.SetArray(facesLV);
    }
    catch (NIERROR _err)
	{
        error = _err;
    }
    catch (...)
	{
        error = NI_ERR_OCV_USER;
    }
    ProcessNIError(error, errorHandle);
}
