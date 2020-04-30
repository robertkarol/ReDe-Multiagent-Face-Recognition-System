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
static CascadeClassifier faceCheckerCascadeClassifier;

void Convert(const Rect& rect, LV_Rect& lvRect)
{
    lvRect.left = rect.x;
    lvRect.top = rect.y;
    lvRect.right = rect.x + rect.width - 1;
    lvRect.bottom = rect.y + rect.height - 1;
    lvRect.angle = 0;
}

Mat Mat2MatGray(Mat colorMat)
{
	Mat matGray;

	if (colorMat.type() != CV_8UC1)
	{
		cvtColor(colorMat, matGray, COLOR_BGR2GRAY);
	}
	else
	{
		matGray = colorMat.clone();
	}
	equalizeHist(matGray, matGray);

	return matGray;
}

EXTERN_C void NI_EXPORT NIVisOpenCV_LoadClassifier(	const char * faceCascadePath,
													const char * faceCheckerCascadePath,
													NIErrorHandle errorHandle)
{
	ReturnOnPreviousError(errorHandle);
	NIERROR error = NI_ERR_SUCCESS;
	String faceCascadeString = faceCascadePath;
	if (!faceCascadeClassifier.load(faceCascadeString))
	{
		error = NI_ERR_OCV_USER;
	}
	ProcessNIError(error, errorHandle);
	String faceCheckerCascadeString = faceCheckerCascadePath;
	if (!faceCheckerCascadeClassifier.load(faceCheckerCascadeString))
	{
		error = NI_ERR_OCV_USER;
	}
	ProcessNIError(error, errorHandle);
}

void FillExistingFaces(NIArray1D<LV_Rect>& facesRect, Mat& matGray)
{
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
}

vector<LV_Rect> TryFilterOutFalsePositives(vector<Rect> faces, Mat& matGray, double scaleFactor, int minNeighbors)
{
	LV_Rect faceLV;
	vector<LV_Rect> facesLV;

	for (vector<Rect>::iterator face = faces.begin(); face != faces.end(); ++face)
	{
		vector<Rect> reallyFaces;
		auto faceToTest = matGray(*face);
		faceCheckerCascadeClassifier.detectMultiScale(faceToTest, reallyFaces, scaleFactor, minNeighbors, 0 | CASCADE_SCALE_IMAGE);
		if (reallyFaces.size() == 0) continue;
		Convert(*face, faceLV);
		facesLV.push_back(faceLV);
	}

	return facesLV;
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
        Mat sourceMat;
        ThrowNIError(source.ImageToMat(sourceMat));
		auto matGray = Mat2MatGray(sourceMat);

        NIArray1D<LV_Rect> facesRect(facesRectLV);
		FillExistingFaces(facesRect, matGray);

        vector<Rect> faces;
        faceCascadeClassifier.detectMultiScale(matGray, faces, scaleFactor, minNeighbors, 
			0 | CASCADE_SCALE_IMAGE, Size(minWidth, minWidth), Size(maxWidth, maxWidth));
        
		vector<LV_Rect> facesLV;
		if (faces.size()) 
		{
			facesLV = TryFilterOutFalsePositives(faces, matGray, scaleFactor, minNeighbors);
		}
        facesRect.SetArray(facesLV);

    }
    catch (NIERROR& _err)
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
	auto matGray = Mat2MatGray(sourceMat);

	vector<Rect> faces;
	faceCheckerCascadeClassifier.detectMultiScale(matGray, faces, scaleFactor, minNeighbors,
		0 | CASCADE_SCALE_IMAGE, Size(matGray.cols / 4, matGray.rows / 4), Size(matGray.cols, matGray.rows));
	ProcessNIError(error, errorHandle);
	*containsFace = faces.size() > 0;
}
