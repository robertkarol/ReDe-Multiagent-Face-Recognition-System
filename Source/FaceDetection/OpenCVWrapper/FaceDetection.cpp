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
#include <opencv2/imgproc/imgproc.hpp>
#include "FaceDetection.h"
using namespace std;
using namespace cv;

typedef cv::Point2f PointFloat;
typedef cv::Point2d PointDouble;

static CascadeClassifier faceCascadeClassifier;
static CascadeClassifier lbpCascadeClassifier;
static CascadeClassifier eyesCascadeClassfier;
static bool bEyeClassifierLoaded = false;

void Convert(const Rect& rect, LV_Rect& lvRect){
    lvRect.left = rect.x;
    lvRect.top = rect.y;
    lvRect.right = rect.x + rect.width - 1;
    lvRect.bottom = rect.y + rect.height - 1;
    lvRect.angle = 0;
}

EXTERN_C void NI_EXPORT NIVisOpenCV_LoadClassifier(	const char * faceCascadePath,
													const char * eyesCascadePath,
													NIErrorHandle errorHandle)
{
	ReturnOnPreviousError(errorHandle);
	String faceCascadeString = faceCascadePath;
	if (!faceCascadeClassifier.load(faceCascadeString))
	{
		return;
	}
	String lbpCascadeString = "E:\\Licenta\\trip-validation-system\\Resources\\lbpcascade_frontalface.xml";
	if (!lbpCascadeClassifier.load(lbpCascadeString))
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

EXTERN_C void NI_EXPORT NIVisOpenCV_DetectFaces(NIImageHandle sourceHandle, 
												NIArrayHandle facesRectLV, 
												NIArrayHandle eyesRectLV, 
												int minWidth, 
												int maxWidth, 
												double scaleFactor,
												int minNeighbors,
												NIErrorHandle errorHandle)
{
    
    ReturnOnPreviousError(errorHandle);
    NIERROR error = NI_ERR_SUCCESS;
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
		
		if (sourceMat.type() != CV_8UC1) 
		{
			cvtColor(sourceMat, matGray, COLOR_BGR2GRAY);
		}
		else 
		{
			matGray = sourceMat.clone();
		}

		if (facesRect.size > 0)
		{
			vector<LV_Rect> existingFaces;
			facesRect.ToVector(existingFaces);

			for (const auto& face : existingFaces)
			{
				auto topLeft = Point(face.left, face.top);
				auto bottomRight = Point(face.right, face.bottom);
				rectangle(matGray, topLeft, bottomRight, cv::Scalar(0, 255, 0), cv::FILLED);
			}
		}
		//rectangle(sourceMat, Point(100, 100), Point(50, 50), cv::Scalar(0, 255, 0), cv::FILLED);
		//imwrite("wtfdidiget.jpg", sourceMat);
		//Equalize image
        equalizeHist(matGray, matGray);
		
		//Detect faces
        faceCascadeClassifier.detectMultiScale(matGray, faces, scaleFactor, minNeighbors, 
			0 | CASCADE_SCALE_IMAGE, Size(minWidth, minWidth), Size(maxWidth, maxWidth));
        
		//facesLV.resize(faces.size());
		LV_Rect faceLV;
        vector<LV_Rect>::iterator fLV = facesLV.begin();
		if (faces.size()) 
		{
			for (vector<Rect>::iterator f = faces.begin(); f != faces.end(); f++) 
			{
				vector<Rect> reallyFaces;
				auto faceToTest = matGray(*f);
				lbpCascadeClassifier.detectMultiScale(faceToTest, reallyFaces, scaleFactor, minNeighbors, 0 | CASCADE_SCALE_IMAGE);
				if (reallyFaces.size() == 0) continue;
				Convert(*f, faceLV);
				facesLV.push_back(faceLV);
			}
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

void NI_EXPORT NIVisOpenCV_ContainsFace(NIImageHandle sourceHandle, 
										double scaleFactor,
										int minNeighbors,
										int8_t* containsFace,
										NIErrorHandle errorHandle)
{
	ReturnOnPreviousError(errorHandle);
	NIERROR error = NI_ERR_SUCCESS;
	Mat sourceMat;
	NIImage source(sourceHandle);
	ThrowNIError(source.ImageToMat(sourceMat));
	Mat matGray;
	if (sourceMat.type() != CV_8UC1)
	{
		cvtColor(sourceMat, matGray, COLOR_BGR2GRAY);
	}
	else
	{
		matGray = sourceMat.clone();
	}
	equalizeHist(matGray, matGray);
	vector<Rect> faces;
	lbpCascadeClassifier.detectMultiScale(matGray, faces, scaleFactor, minNeighbors,
		0 | CASCADE_SCALE_IMAGE, Size(matGray.cols / 4, matGray.rows / 4), Size(matGray.cols, matGray.rows));
	ProcessNIError(error, errorHandle);
	*containsFace = faces.size() > 0;
}
