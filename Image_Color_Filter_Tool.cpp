// Image_Color_Filter_Tool.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <io.h>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Color_Filter.h"

using namespace cv;
using namespace std;

int calc_lut1d(Mat &origin, Mat &target, Mat &lut1d)
{
	int i, j, offset;
	Mat bgrao, bgrat;
	Mat lut(LUT_ROWS, LUT_COLS, CV_8UC4, cv::Scalar::all(0));
	Mat lut_channels[4];
	vector<vector<unsigned char> > vec0(256);
	vector<vector<unsigned char> > vec1(256);
	vector<vector<unsigned char> > vec2(256);
	vector<vector<unsigned char> > vec3(256);
	if (origin.channels() == 3)
		cvtColor(origin, bgrao, COLOR_BGR2BGRA);
	else
		bgrao = origin;
	if (target.channels() == 3)
		cvtColor(target, bgrat, COLOR_BGR2BGRA);
	else
		bgrat = target;
	int height = bgrao.rows;
	int width = bgrao.cols;
	int channels = bgrao.channels();
	int stride = bgrao.step.buf[0];
	unsigned char * pSrc = bgrao.data;
	unsigned char * pDst = bgrat.data;

	///////////// fill zero pixel size
	offset = stride - (width * channels);
	//////////////////////////////////
	//////statistics b g r a component
	cv::split(lut, lut_channels);
	for (j = 0; j < height; j++)
	{
		for (i = 0; i < width; i++)
		{
			vec0[pSrc[0]].push_back(pDst[0]);
			vec1[pSrc[1]].push_back(pDst[1]);
			vec2[pSrc[2]].push_back(pDst[2]);
			vec3[pSrc[3]].push_back(pDst[3]);
			pSrc += channels;
			pDst += channels;
		}
		pSrc += offset;
		pDst += offset;
	}
	////////////////////////////////////////
	///////calculate b g r a component means
	Calculate_Rgba(vec0, 256, lut_channels[0]);
	Calculate_Rgba(vec1, 256, lut_channels[1]);
	Calculate_Rgba(vec2, 256, lut_channels[2]);
	Calculate_Rgba(vec3, 256, lut_channels[3]);
	////////////////////////////////////////
	/////////connect empty pxiel
	Connect_Lut(vec0, 256, lut_channels[0]);
	Connect_Lut(vec1, 256, lut_channels[1]);
	Connect_Lut(vec2, 256, lut_channels[2]);
	Connect_Lut(vec3, 256, lut_channels[3]);
	////////////////////////////////////////
	/////////Fliter lut curve
	for (i = 0; i < 256; i++)
	{
		static vector<unsigned char> buf0;
		static vector<unsigned char> buf1;
		static vector<unsigned char> buf2;
		static vector<unsigned char> buf3;
		if (i < FLITER_LEN)
		{
			buf0.push_back(lut_channels[0].data[i]);
			buf1.push_back(lut_channels[1].data[i]);
			buf2.push_back(lut_channels[2].data[i]);
			buf3.push_back(lut_channels[3].data[i]);
			continue;
		}
		else
		{
			buf0.erase(buf0.begin());
			buf0.push_back(lut_channels[0].data[i]);
			buf1.erase(buf1.begin());
			buf1.push_back(lut_channels[1].data[i]);
			buf2.erase(buf2.begin());
			buf2.push_back(lut_channels[2].data[i]);
			buf3.erase(buf3.begin());
			buf3.push_back(lut_channels[3].data[i]);
		}
		lut_channels[0].data[i - int(FLITER_LEN / 2)] = Filter_Lut(buf0, FLITER_LEN);
		lut_channels[1].data[i - int(FLITER_LEN / 2)] = Filter_Lut(buf1, FLITER_LEN);
		lut_channels[2].data[i - int(FLITER_LEN / 2)] = Filter_Lut(buf2, FLITER_LEN);
		lut_channels[3].data[i - int(FLITER_LEN / 2)] = Filter_Lut(buf3, FLITER_LEN);
	}
	for (i = 1; i < 50; i++)
	{
		lut_channels[0].row(0).copyTo(lut_channels[0].row(i));
		lut_channels[1].row(0).copyTo(lut_channels[1].row(i));
		lut_channels[2].row(0).copyTo(lut_channels[2].row(i));
		lut_channels[3].row(0).copyTo(lut_channels[3].row(i));
	}
	cv::merge(lut_channels, 4, lut1d);

	return 0;
}
int calc_matrix(Mat &origin, Mat &target, Mat &matrix)
{
	int i;
	Mat channelo[4];
	Mat channelt[4];
	Mat orig32, targ32, bgrao, bgrat, re_orig, re_targ, inv;
	if (origin.channels() == 3)
		cvtColor(origin, bgrao, COLOR_BGR2BGRA);
	else
		bgrao = origin;
	if (target.channels() == 3)
		cvtColor(target, bgrat, COLOR_BGR2BGRA);
	else
		bgrat = target;
	split(bgrao, channelo);
	split(bgrat, channelt);

	for (i = 1; i < bgrao.channels(); i++)
	{
		channelo[0].push_back(channelo[i]);
		channelt[0].push_back(channelt[i]);
	}

	re_orig = channelo[0].reshape(1, 4);
	re_targ = channelt[0].reshape(1, 4);
	Mat add1(1, re_orig.cols, CV_8UC1, cv::Scalar::all(1));
	re_orig.push_back(add1);

	re_orig.convertTo(orig32, CV_32FC1);
	re_targ.convertTo(targ32, CV_32FC1);

	inv = orig32.inv(DECOMP_SVD);
	matrix = targ32 * inv;
	return 0;
}

int predict_lut1d(Mat &origin, Mat &result, Mat &lut1d)
{
	Mat bgrao;
	Mat lut_channels[4];
	if (origin.channels() < 4)
		cvtColor(origin, bgrao, COLOR_BGR2BGRA);
	else
		bgrao = origin;
	Mat channels_test[4];
	Mat dest[4];
	Mat lut0, lut1, lut2, lut3;
	cv::split(bgrao, channels_test);
	cv::split(lut1d, lut_channels);
	lut0 = lut_channels[0].rowRange(0, 1).clone();
	lut1 = lut_channels[1].rowRange(0, 1).clone();
	lut2 = lut_channels[2].rowRange(0, 1).clone();
	lut3 = lut_channels[3].rowRange(0, 1).clone();
	LUT(channels_test[0], lut0, dest[0]);
	LUT(channels_test[1], lut1, dest[1]);
	LUT(channels_test[2], lut2, dest[2]);
	LUT(channels_test[3], lut3, dest[3]);
	cv::merge(dest, 4, result);
	return 0;
}
int predict_matrix(Mat &origin, Mat &result, Mat &matrix)
{
	int i;
	Mat bgrao, test32, gert;
	Mat channelw[4];
	Mat channelr[4];
	if (origin.channels() < 4)
		cvtColor(origin, bgrao, COLOR_BGR2BGRA);
	else
		bgrao = origin;
	split(bgrao, channelw);
	for (i = 1; i < bgrao.channels(); i++)
		channelw[0].push_back(channelw[i]);

	Mat re_test = channelw[0].reshape(1, 4);
	Mat add2(1, re_test.cols, CV_8UC1, cv::Scalar::all(1));
	re_test.push_back(add2);
	re_test.convertTo(test32, CV_32FC1);
	Mat inter = matrix * test32;
	channelr[0] = inter.rowRange(0, 1).clone();
	channelr[1] = inter.rowRange(1, 2).clone();
	channelr[2] = inter.rowRange(2, 3).clone();
	channelr[3] = inter.rowRange(3, 4).clone();
	merge(channelr, 4, gert);

	Mat re_result = gert.reshape(4, origin.rows);
	re_result.convertTo(result, CV_8UC4);
	return 0;
}

int main()
{
	int i, j, frameIndex;
	double meanDiff;
	string train_imageIn_folder_path = "./Filter-1dLut/origin";
	string train_imageOut_folder_path = "./Filter-1dLut/target/japanese/clear";
	string test_imageIn_folder_path = "./Filter-1dLut/origin";
	string test_imageOut_folder_path = "./Filter-1dLut/result/japanese/clear";

	Mat origin;
	Mat target;
	Mat result;
	Mat tmp;
	Mat aveDiff;
	Mat aveLut1dCurve(LUT_ROWS, LUT_COLS, CV_32FC4, cv::Scalar::all(0));
	Mat aveMatrix4x5(4, 5, CV_32FC1, cv::Scalar::all(0));
	Mat matrix_4x5;
	Mat lut1d_curve;

	/* load image list of the train folder here */
	// imageList
	std::vector<std::string> imageList_origin, imageList_target, imageList_test;
	std::vector<std::string> imageNames_origin, imageNames_target, imageNames_test;

	listFiles(train_imageIn_folder_path.c_str(), imageList_origin, imageNames_origin);
	listFiles(train_imageOut_folder_path.c_str(), imageList_target, imageNames_target);
	/* define regression method */
	stageKey method[] = { LUT1D, MATRIX };// MATRIX, LUT1D
	/* do regression for each stage */
	for (frameIndex = 0; frameIndex < imageList_origin.size(); frameIndex ++)
	{
		/* read image data to mat */
		// origin, target
		origin = imread(imageList_origin[frameIndex] + imageNames_origin[frameIndex], CV_LOAD_IMAGE_UNCHANGED);
		target = imread(imageList_target[frameIndex] + imageNames_origin[frameIndex], CV_LOAD_IMAGE_UNCHANGED);
		if (origin.empty() || target.empty())
		{
			printf("reading image error");
			return 1;
		}
		for (i = 0; i < sizeof(method) / sizeof(method[0]); i++)
		{
			stageKey key = method[i];
			switch (key)
			{
			case LUT1D:
				calc_lut1d(origin, target, lut1d_curve);
				predict_lut1d(origin, result, lut1d_curve);
				origin = result.clone();
				lut1d_curve.convertTo(tmp, CV_32FC4);
				aveLut1dCurve += tmp;
				break;
			case MATRIX:
				calc_matrix(origin, target, matrix_4x5);
				predict_matrix(origin, result, matrix_4x5);
				origin = result.clone();
				aveMatrix4x5 += matrix_4x5;
				break;
			default:
				break;
			}
		}

	}

	/* average regression matrix and lut1d */
	aveMatrix4x5 /= frameIndex;
	aveLut1dCurve /= frameIndex;

	/* save CV_8UC4 aveLut1dCurve and CV_32UC4 aveMatrix4x5 */
	Mat saveCurve;
	aveLut1dCurve.convertTo(saveCurve, CV_8UC4);

	FileStorage fs(".\\aveLut1dCurve.xml", FileStorage::WRITE);
	fs << "aveLut1dCurve" << saveCurve;
	fs.release();
	fs.open(".\\aveMatrix4x5.xml", FileStorage::WRITE);
	fs << "aveMatrix4x5" << aveMatrix4x5;
	fs.release();
#if 0
	fs.open(".\\sweet\\lut\\aveLut1dCurve.xml", FileStorage::READ);
	fs["aveLut1dCurve"] >> saveCurve;
	fs.open(".\\sweet\\lut\\aveMatrix4x5.xml", FileStorage::READ);
	fs["aveMatrix4x5"] >> aveMatrix4x5;
#endif
	/* test image */
	listFiles(test_imageIn_folder_path.c_str(), imageList_test, imageNames_test);
	for (frameIndex = 0; frameIndex < imageList_test.size(); frameIndex++)
	{
		origin = imread(imageList_test[frameIndex] + imageNames_test[frameIndex], CV_LOAD_IMAGE_UNCHANGED);
		if (origin.empty())
		{
			printf("reading image error");
			return 1;
		}
		Mat diffIn = origin.clone();
		for (i = 0; i < sizeof(method) / sizeof(method[0]); i++)
		{
			stageKey key = method[i];
			switch (key)
			{
			case LUT1D:
				predict_lut1d(origin, result, saveCurve);
				origin = result.clone();
				//imwrite(test_imageOut_folder_path + "/" + imageNames_test[frameIndex], result);
				break;
			case MATRIX:
				predict_matrix(origin, result, aveMatrix4x5);
				origin = result.clone();
				//imwrite(test_imageOut_folder_path + "/" + imageNames_test[frameIndex], result);
				break;
			default:
				break;
			}
		}

		imwrite(test_imageOut_folder_path + "/" + imageNames_test[frameIndex], result);
	}
    return 0;
}

