#pragma once

#define NEW_CHANGE 1
#define BY_PASS_MEAN 2
#define BY_PASS_COLLECT (((BY_PASS_MEAN)*2)+1)
#define FLITER_LEN 5
#define LUT_COLS 256
#define LUT_ROWS 50
#define CALU_LUT_3D 0
#define RESTART 1

/* METHOD FOR REGRESSION */
typedef enum {
	LUT1D = 0,
	MATRIX
}stageKey;


uchar Filter_Lut(std::vector<uchar> &buf_in, int n);

void Connect_Lut(std::vector<std::vector<uchar> > &invec, int size, cv::Mat &outvec);

void Calculate_Rgba(std::vector<std::vector<uchar> > &invec, int size, cv::Mat &outvec);

void listFiles(const char * dir, std::vector<std::string>& files, std::vector<std::string>& files_names);

int Color_1dLut_Filter(std::string o_path, std::string t_path, std::string s_path);

int Color_3dLut_Filter(std::string o_path, std::string t_path, std::string s_path);

int Color_Matrix_Filter(std::string o_path, std::string t_path, std::string s_path);




