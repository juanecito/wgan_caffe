/*
 * img_viewer.cpp
 *
 *  Created on: 24 jul. 2017
 *      Author: juan
 */



#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

////////////////////////////////////////////////////////////////////////////////
void show_direct_grid_img_CV_32FC3(const cv::Mat& img)
{
	cv::imshow("cifar10_generator", img);
	cv::waitKey();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void read_grid_img_CV_32FC3(const std::string& file_name, cv::Mat& img)
{
	cv::FileStorage fs(file_name.c_str(), cv::FileStorage::READ);
	fs["grid_img"] >> img;
	fs.release();
}

int main(int argc, char** ppcargv)
{
	if (argc != 2) return 1;

	const std::string& file_name = std::string(ppcargv[1]);

	cv::Mat img;

	read_grid_img_CV_32FC3(file_name, img);

	show_direct_grid_img_CV_32FC3(img);

	return 0;
}
