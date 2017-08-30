/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-  */
/*
 * CLFWFaceDatabase.cpp
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

/** @file CLFWFaceDatabase.cpp.cpp
 * @author Juan Maria Gomez Lopez <juanecitorr@gmail.com>
 * @date 31 Jul 2017
 */

#include <dirent.h>
#include <sys/stat.h>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "CLFWFaceDatabase.hpp"

////////////////////////////////////////////////////////////////////////////////
CLFWFaceDatabase::CLFWFaceDatabase()
{
	is_loaded_ = false;
	count_imgs_ = 0;
	path_.clear();
}

////////////////////////////////////////////////////////////////////////////////
bool folderExists(const std::string& file_name)
{
	struct stat st;
	if (stat(file_name.c_str(), &st) != 0)
	{
		return false;
	}
	else
	{
		if (!S_ISDIR(st.st_mode))
		{
			return false;
		}
	}
	return true;
}

////////////////////////////////////////////////////////////////////////////////
bool get_jpg_files_from_path(const std::string& folder_path,
									std::vector<std::string>& jpg_file_names)
{
	struct dirent *drnt;
	DIR* dir = opendir(folder_path.c_str());
	if (dir == nullptr) return false;
	drnt = readdir(dir);

	const static std::string ext = ".jpg";
	std::string file_name = "";

	while(drnt != nullptr)
	{
		if (drnt->d_type == DT_REG)
		{
			file_name = std::string(drnt->d_name);
			if (file_name.substr(file_name.length() - 4, 4).compare(ext) == 0)
			{
				jpg_file_names.push_back(folder_path + std::string("/") + file_name);
			}
		}
		else if (drnt->d_type == DT_DIR && drnt->d_name[0] != '.')
		{
			get_jpg_files_from_path(
				folder_path + std::string("/") + std::string(drnt->d_name),
				jpg_file_names);
		}

		drnt = readdir(dir);
	}

	closedir(dir);

	return true;
}

////////////////////////////////////////////////////////////////////////////////
bool load_from_jpg(const std::string& file_path,
								struct S_LFW_FaceDB_img_rgb<uint8_t>* img_rgb,
								struct S_LFW_FaceDB_img<uint8_t>* img)
{
	cv::Mat ini_tmp_img;
	ini_tmp_img = cv::imread(file_path, CV_LOAD_IMAGE_COLOR);

	if(!ini_tmp_img.data )                              // Check for invalid input
	{
		std::cout <<  "Could not open or find the image" << std::endl;
		return false;
	}
	else
	{
		// resize img a 64 x 64
		cv::Size size(64, 64);
		cv::Mat tmp_img(64, 64, CV_8UC3);
		cv::resize(ini_tmp_img, tmp_img, size, 64.0 / 250.0,
						64.0 / 250.0, CV_INTER_LINEAR);


		unsigned int img_count = tmp_img.channels() * tmp_img.rows * tmp_img.cols;
		memcpy(img_rgb->rgb_, tmp_img.data, img_count * sizeof (uint8_t));

		std::vector<cv::Mat> channels(3);
		// split img:
		cv::split(tmp_img, channels);

		// get the channels (dont forget they follow BGR order in OpenCV)
		memcpy(img->red_channel_, channels.at(0).data,
				channels.at(0).rows * channels.at(0).cols * sizeof(uint8_t));

		memcpy(img->green_channel_, channels.at(1).data,
				channels.at(1).rows * channels.at(1).cols * sizeof(uint8_t));

		memcpy(img->blue_channel_, channels.at(2).data,
				channels.at(2).rows * channels.at(2).cols * sizeof(uint8_t));
	}

	return true;
}

////////////////////////////////////////////////////////////////////////////////
void CLFWFaceDatabase::load()
{
	if (path_.empty() || !folderExists(path_)) return;

	std::vector<std::string> jpg_file_names;
	jpg_file_names.clear();
	get_jpg_files_from_path(path_, jpg_file_names);

	data_.reset(new struct S_LFW_FaceDB_img<uint8_t> [jpg_file_names.size()]);
	rgb_data_.reset(new struct S_LFW_FaceDB_img_rgb<uint8_t> [jpg_file_names.size()]);

	this->count_imgs_ = jpg_file_names.size();
	unsigned int index = 0;
	for (auto it : jpg_file_names)
	{
		// std::cout << it << std::endl;
		load_from_jpg(it, &(rgb_data_.get()[index]), &(data_.get()[index]));

		index++;
	}
}

