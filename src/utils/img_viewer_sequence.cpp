/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-  */
/*
 * img_viewer_sequence.cpp
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

/** @file * img_viewer_sequence.cpp
 * @author Juan Maria Gomez Lopez <juanecitorr@gmail.com>
 * @date 24 Jul 2017
 */

#include <iostream>
#include <string>
#include <algorithm>
#include <functional>
#include <sstream>
#include <iomanip>

#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


////////////////////////////////////////////////////////////////////////////////
void show_direct_grid_img_CV_32FC3(const std::string& name, const cv::Mat& img)
{
	cv::imshow(name.c_str(), img);
	cv::waitKey();
}

bool write_img_CV_32FC3_to_jpeg_file(const char* folder_path,
		const std::string& name, const cv::Mat& img)
{
	std::string jpeg_file_name = "";
	const static std::string ext = ".jpg";
	std::string str_prefix = name.substr(0, 9);
	std::string str_number = name.substr(9, name.length() - 9 - 4);
	int number = atoi(str_number.c_str());
	std::stringstream ss;
	ss << str_prefix << std::setfill('0') << std::setw(5) << number << ext;
	jpeg_file_name = std::string(folder_path) + std::string("/") + ss.str();

	cv::Mat newImage;
	img.convertTo(newImage, CV_8UC3, 255.0);

	std::vector<int> params = {CV_IMWRITE_JPEG_QUALITY, 100};

	cv::imwrite( jpeg_file_name.c_str(), newImage, params);

	return true;
}

////////////////////////////////////////////////////////////////////////////////
void read_grid_img_CV_32FC3(const std::string& file_name, cv::Mat& img)
{
	cv::FileStorage fs(file_name.c_str(), cv::FileStorage::READ);
	fs["grid_img"] >> img;
	fs.release();
}

////////////////////////////////////////////////////////////////////////////////
bool get_yml_files_from_path(const std::string& folder_path,
									std::vector<std::string>& yml_file_names)
{
	struct dirent *drnt;

	DIR* dir = opendir(folder_path.c_str());

	if (dir == nullptr) return false;

	drnt = readdir(dir);

	const std::string ext = ".yml";
	std::string file_name = "";

	yml_file_names.clear();

	while(drnt != nullptr)
	{
		if (drnt->d_type == DT_REG)
		{
			file_name = std::string(drnt->d_name);
			if (file_name.substr(file_name.length() - 4, 4).compare(ext) == 0)
			{
				yml_file_names.push_back(file_name);
			}
		}

		drnt = readdir(dir);
	}

	closedir(dir);

	return true;
}

////////////////////////////////////////////////////////////////////////////////
bool create_mat_from_yml(const char* folder_path,
						const std::vector<std::string>& yml_file_names,
										std::map<std::string, cv::Mat>& mats)
{
	mats.clear();

	for (auto it : yml_file_names)
	{
		cv::Mat img;
		read_grid_img_CV_32FC3(
					std::string(folder_path) + std::string("/") + it, img);

		mats.insert(std::pair<std::string, cv::Mat>(it, img));

	}

	return true;
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** ppcargv)
{
	if (argc != 2) return 1;

	std::vector<std::string> yml_file_names;
	yml_file_names.clear();

	auto order_yml_file_greater = [] (const std::string& a, const std::string& b)
			{
				std::string str_number_a = a.substr(9, a.length() - 9 - 4);
				std::string str_number_b = b.substr(9, b.length() - 9 - 4);

				int number_a = atoi(str_number_a.c_str());
				int number_b = atoi(str_number_b.c_str());

				return number_a < number_b;
			};

	if (get_yml_files_from_path(ppcargv[1], yml_file_names))
	{
		std::sort(yml_file_names.begin(), yml_file_names.end(), order_yml_file_greater);

		std::map<std::string, cv::Mat> mats;
		mats.clear();

		create_mat_from_yml(ppcargv[1], yml_file_names, mats);

		for (auto it : yml_file_names)
		{
			auto it_map = mats.find(it);
			//show_direct_grid_img_CV_32FC3(it_map->first, it_map->second);
			write_img_CV_32FC3_to_jpeg_file(ppcargv[1], it_map->first, it_map->second);
		}
	}

	return 0;
}
