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
void read_grid_img_CV_32FC3(const std::string& file_name,
		unsigned int img_width, unsigned int img_height,
		unsigned int channels, unsigned int grid_width, unsigned int grid_height, cv::Mat& img)
{
	unsigned int img_count = grid_height * grid_width;
	unsigned int img_size_per_channel = img_height * img_width;
	unsigned int img_size = channels * img_size_per_channel;

	unsigned int grid_img_count = img_count * img_size;

	//float* tranf_img_data = new float [grid_img_count];

	//const cv::Mat img_tmp(img_width * grid_width, img_height * grid_height, CV_32FC3, tranf_img_data);

	cv::FileStorage fs(file_name.c_str(), cv::FileStorage::READ);
	fs["grid_img"] >> img;
	fs.release();

	//img = img_tmp;

	//delete[] tranf_img_data;
}

int main(int argc, char** ppcargv)
{
	if (argc != 2) return 1;

	const std::string& file_name = std::string(ppcargv[1]);

	cv::Mat img;

	read_grid_img_CV_32FC3(file_name, 64, 64, 3, 8, 8, img);

	show_direct_grid_img_CV_32FC3(img);

	return 0;
}
