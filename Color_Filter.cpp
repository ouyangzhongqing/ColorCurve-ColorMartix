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
void listFiles(const char * dir, vector<string>& files, vector<string>& files_names)
{
	char dirNew[200];
	strcpy(dirNew, dir);
	strcat(dirNew, "\\*.*");    // 在目录后面加上"\\*.*"进行第一次搜索
	intptr_t handle;
	_finddata_t findData;
	handle = _findfirst(dirNew, &findData);
	if (handle == -1)        // 检查是否成功
		return;
	do
	{
		if (findData.attrib & _A_SUBDIR)
		{
			if (strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0)
				continue;
			cout << findData.name << "\t<dir>\n";
			// 在目录后面加上"\\"和搜索到的目录名进行下一次搜索
			strcpy(dirNew, dir);
			strcat(dirNew, "\\");
			strcat(dirNew, findData.name);
			listFiles(dirNew, files, files_names);
		}
		else
		{
			files.push_back(string(dir).append("\\"));//.append(findData.name)
			files_names.push_back(findData.name);
		}
		cout << findData.name << "\t" << findData.size << " bytes.\n";
	} while (_findnext(handle, &findData) == 0);
	_findclose(handle);    // 关闭搜索句柄
}
uchar Filter_Lut(vector<unsigned char> &buf_in, int n)
{
	int i, j;
	char filter_temp;
	int filter_sum = 0;
	////sort
	sort(buf_in.begin(), buf_in.end());
	// 去除最大最小极值后求平均
	for (i = 1; i < n - 1; i++) filter_sum += buf_in[i];
	return (uchar)(filter_sum / (n - 2));
}
void Connect_Lut(vector<vector<unsigned char> > &invec, int size, Mat &outvec)
{
	int i;
	vector<int> pixel_empty;

	for (i = 0; i < 256; i++)
	{
		int begin, end, l;
		float slope;
		if (invec[i].size() <= BY_PASS_COLLECT)
		{
			pixel_empty.push_back(i);
			if (i < 255)
				continue;
		}
		if (pixel_empty.size() > 0)
		{
			begin = max(pixel_empty[0] - 1, 0);
			end = min(pixel_empty[pixel_empty.size() - 1] + 1, 255);
			uchar tmp = outvec.data[begin];
			slope = (float)(outvec.data[end] - outvec.data[begin]) / (float)pixel_empty.size();
			l = 1;
			for (vector<int>::iterator it = pixel_empty.begin(); it != pixel_empty.end(); it++)
			{
				if (begin == 0)
				{
					outvec.data[*it] = (uchar)(slope*(l - 1) + tmp);
					l++;
				}
				else
				{
					outvec.data[*it] = (uchar)(slope*l + tmp);
					l++;
				}
			}
			pixel_empty.clear();
		}
	}
}
void Calculate_Rgba(vector<vector<unsigned char> > &invec, int size, Mat &outvec)
{
	int i;
	for (i = 0; i < 256; i++)
	{
		if (invec[i].size() <= BY_PASS_COLLECT)
		{
			outvec.data[i] = i;
		}
		else
		{
			sort(invec[i].begin(), invec[i].end());
			double sum1 = accumulate(begin(invec[i]) + BY_PASS_MEAN, end(invec[i]) - BY_PASS_MEAN, 0.0);
			double mean1 = sum1 / (invec[i].size() - BY_PASS_MEAN * 2);
			outvec.data[i] = (uchar)mean1;
		}
	}
}

int Color_3dLut_Filter(string o_path, string t_path, string s_path)
{
	return 0;
}

int Color_1dLut_Filter(string o_path, string t_path, string s_path)
{
	vector<vector<unsigned char> > vec0(256);
	vector<vector<unsigned char> > vec1(256);
	vector<vector<unsigned char> > vec2(256);
	vector<vector<unsigned char> > vec3(256);
	int  m, i, j, r, g, b, a, offset, pos, nx, ny, k;
	Mat orig, targ, bgrao, bgrat;
	Mat lut(LUT_ROWS, LUT_COLS, CV_8UC4, cv::Scalar::all(0));
	Mat lut_channels[4];
	Mat lut_dest;
	vector<Mat> lut_collect;
	//vector<int> pixel_empty;

	string orig_path, targ_path, save_path;
	vector<string> files_origin, files_target;
	vector<string> file_names_origin, file_names_target;
	////reading list of target image, and read image data in for()
	listFiles(o_path.c_str(), files_origin, file_names_origin);
	listFiles(t_path.c_str(), files_target, file_names_target);
#if RESTART
	for (int m = 0; m < files_origin.size(); m++)
	{
		Mat showlut(LUT_COLS, LUT_COLS, CV_8UC4, cv::Scalar::all(255));
		orig_path = files_origin[m] + file_names_origin[m];
		orig = imread(orig_path, CV_LOAD_IMAGE_UNCHANGED);
		string tmps1 = file_names_origin[m];
		string tmps2 = tmps1.replace(tmps1.length() - 4, 4, file_names_target[m].substr(3, 9));
		targ_path = files_target[m] + tmps2;
		targ = imread(targ_path, CV_LOAD_IMAGE_UNCHANGED);
		if (orig.empty() || targ.empty())
		{
			printf("reading image error");
			return 1;
		}
		save_path = s_path + string("/") + tmps2;

		if (orig.channels() == 3)
			cvtColor(orig, bgrao, COLOR_BGR2BGRA);
		else
			bgrao = orig;
		if (targ.channels() == 3)
			cvtColor(targ, bgrat, COLOR_BGR2BGRA);
		else
			bgrat = targ;

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
				b = pSrc[0];
				g = pSrc[1];
				r = pSrc[2];
				a = pSrc[3];
				vec0[b].push_back(pDst[0]);
				vec1[g].push_back(pDst[1]);
				vec2[r].push_back(pDst[2]);
				vec3[a].push_back(pDst[3]);
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
		//////////////////////////////////////////
		///////draw copy and merge to lut(50,256,3)
		for (i = 0; i < 256; i++)
		{
			circle(showlut, cv::Point(i, 255 - (int)lut_channels[0].data[i]), 0.001, Scalar(255, 0, 0), -1); //第五个参数我设为-1，表明这是个实点。
			circle(showlut, cv::Point(i, 255 - (int)lut_channels[1].data[i]), 0.001, Scalar(0, 255, 0), -1); //第五个参数我设为-1，表明这是个实点。
			circle(showlut, cv::Point(i, 255 - (int)lut_channels[2].data[i]), 0.001, Scalar(0, 0, 255), -1); //第五个参数我设为-1，表明这是个实点。
			circle(showlut, cv::Point(i, 255 - (int)lut_channels[3].data[i]), 0.001, Scalar(0, 0, 0), -1); //第五个参数我设为-1，表明这是个实点。
		}
		for (i = 1; i < 50; i++)
		{
			lut_channels[0].row(0).copyTo(lut_channels[0].row(i));
			lut_channels[1].row(0).copyTo(lut_channels[1].row(i));
			lut_channels[2].row(0).copyTo(lut_channels[2].row(i));
			lut_channels[3].row(0).copyTo(lut_channels[3].row(i));
		}
		cv::merge(lut_channels, 4, lut_dest);
		Mat tmp_dest;
		lut_dest.convertTo(tmp_dest, CV_32FC4);
		lut_collect.push_back(tmp_dest);
	}
	Mat sum_lut = lut_collect.front();
	for (i = 1; i < lut_collect.size(); i++)
	{
		sum_lut = sum_lut + lut_collect[i];
	}
	Mat mean_lut = sum_lut / lut_collect.size();
	Mat mean_u8c4;
	mean_lut.convertTo(mean_u8c4, CV_8UC4);
	FileStorage fs(".\\vocabulary.xml", FileStorage::WRITE);
	fs << "vocabulary" << mean_u8c4;
	fs.release();
#else
FileStorage fs(".\\vocabulary.xml", FileStorage::READ);
fs["vocabulary"] >> mean_u8c4;
#endif
		/////////////////////////
		////////testing code
		////////////////////////
	for (int m = 0; m < files_origin.size(); m++)
	{
		orig_path = files_origin[m] + file_names_origin[m];
		Mat test = imread(orig_path, CV_LOAD_IMAGE_UNCHANGED);
		if (test.empty())
		{
			printf("test image reading error\n");
			return 1;
		}
		if (test.channels() < 4)
			cvtColor(test, bgrao, COLOR_BGR2BGRA);
		else
			bgrao = test;
		string tmps1 = file_names_origin[m];
		string tmps2 = tmps1.replace(tmps1.length() - 4, 4, file_names_target[m].substr(3, 9));
		save_path = s_path + string("/") + tmps2;
		Mat drt;
		Mat channels_test[4];
		Mat dest[4];
		Mat lut0, lut1, lut2, lut3;
		cv::split(bgrao, channels_test);
		cv::split(mean_u8c4, lut_channels);
		lut0 = lut_channels[0].rowRange(0, 1).clone();
		lut1 = lut_channels[1].rowRange(0, 1).clone();
		lut2 = lut_channels[2].rowRange(0, 1).clone();
		lut3 = lut_channels[3].rowRange(0, 1).clone();
		LUT(channels_test[0], lut0, dest[0]);
		LUT(channels_test[1], lut1, dest[1]);
		LUT(channels_test[2], lut2, dest[2]);
		LUT(channels_test[3], lut3, dest[3]);
		cv::merge(dest, 4, drt);
		cv::imwrite(save_path, drt);
		std::printf("Complete %s\n", file_names_origin[m].c_str());

		/*cvNamedWindow("origin", 0);
		imshow("origin", img_orig);
		cvNamedWindow("lut_dest", 0);
		imshow("lut_dest", lut_dest);
		cvNamedWindow("drt", 0);
		imshow("drt", drt);
		cvNamedWindow("bgra", 0);
		imshow("bgra", showlut);
		printf("wait press key\t");
		cvWaitKey();
		printf("press valid\n");*/
	}
	return 0;
}

int Color_Matrix_Filter(string o_path, string t_path, string s_path)
{
	int  i, j, r, g, b, offset, pos, nx, ny, k;
	Mat channelo[4];
	Mat channelt[4];
	Mat orig32, targ32, bgrao, bgrat, test32, orig, targ, re_orig, re_targ, dest, inv;
	vector<Mat> dest_matrix;
	string orig_path, targ_path, save_path;
	vector<string> files_origin, files_target;
	vector<string> file_names_origin, file_names_target;
	////reading list of target image, and read image data in for()
	listFiles(o_path.c_str(), files_origin, file_names_origin);
	listFiles(t_path.c_str(), files_target, file_names_target);
	for (int m = 0; m < files_origin.size(); m++)
	{
		orig_path = files_origin[m] + file_names_origin[m];
		orig = imread(orig_path, CV_LOAD_IMAGE_UNCHANGED);
		string tmps1 = file_names_origin[m];
		string tmps2 = tmps1.replace(tmps1.length() - 4, 4, file_names_target[m].substr(3, 9));
		targ_path = files_target[m] + tmps2;
		targ = imread(targ_path, CV_LOAD_IMAGE_UNCHANGED);
		if (orig.empty() || targ.empty())
		{
			printf("reading image error");
			return 1;
		}
		save_path = s_path + string("/") + tmps2;

		if (orig.channels() == 3)
			cvtColor(orig, bgrao, COLOR_BGR2BGRA);
		else
			bgrao = orig;
		if (targ.channels() == 3)
			cvtColor(targ, bgrat, COLOR_BGR2BGRA);
		else
			bgrat = targ;
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

		//Mat tmp(4, 5, CV_32F);	//, cv::Scalar::randn
		//cv::randu(tmp, cv::Scalar::all(0), cv::Scalar::all(256));
		//Mat inv_tmp = tmp.inv(DECOMP_SVD);
		inv = orig32.inv(DECOMP_SVD);
		dest = targ32 * inv;
		Mat destmp = dest.clone();
		dest_matrix.push_back(destmp);
	}
	Mat sum_matrix = dest.clone();
	for (i = 0; i < dest_matrix.size() - 1; i++)
	{
		sum_matrix = sum_matrix + dest_matrix[i];
	}
	Mat mean_matrix = sum_matrix / dest_matrix.size();

	////////////// testing code
	for (int m = 0; m < files_origin.size(); m++)
	{
		orig_path = files_origin[m] + file_names_origin[m];
		Mat test = imread(orig_path, CV_LOAD_IMAGE_UNCHANGED);
		string tmps1 = file_names_origin[m];
		string tmps2 = tmps1.replace(tmps1.length() - 4, 4, file_names_target[m].substr(3, 9));
		save_path = s_path + string("/") + tmps2;
		Mat channelw[4];
		Mat channelr[4];
		Mat testa, gert;
		if (test.channels() < 4)
			cvtColor(test, testa, COLOR_BGR2BGRA);
		else
			testa = test;
		split(testa, channelw);
		for (j = 1; j < testa.channels(); j++)
			channelw[0].push_back(channelw[j]);

		Mat re_test = channelw[0].reshape(1, 4);
		Mat add2(1, re_test.cols, CV_8UC1, cv::Scalar::all(1));
		re_test.push_back(add2);
		re_test.convertTo(test32, CV_32FC1);
		Mat result = mean_matrix * test32;
		channelr[0] = result.rowRange(0, 1).clone();
		channelr[1] = result.rowRange(1, 2).clone();
		channelr[2] = result.rowRange(2, 3).clone();
		channelr[3] = result.rowRange(3, 4).clone();
		merge(channelr, 4, gert);

		Mat re_result = gert.reshape(4, test.rows);
		Mat save;
		re_result.convertTo(save, CV_8UC3);
		imwrite(save_path, save);
		printf("Complete %s\n", file_names_origin[m].c_str());

		cvNamedWindow("test", 0);
		imshow("test", test);
		cvNamedWindow("save", 0);
		imshow("save", save);
		cvWaitKey();
	}
	return 0;
}
