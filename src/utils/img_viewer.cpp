/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-  */
/*
 * img_viewer.cpp
 * Copyright (C) 2017 Juan Maria Gomez Lopez <juanecitorr@gmail.com>
 *
 * caffe_network is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * caffe_wgan is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/** @file * img_viewer.cpp
 * @author Juan Maria Gomez Lopez <juanecitorr@gmail.com>
 * @date 24 Jul 2017
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
