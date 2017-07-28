/*
 * img_viewer.cpp
 *
 *  Created on: 24 jul. 2017
 *      Author: juan
 */



#include <iostream>
#include <string>
#include <algorithm>
#include <functional>

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
bool create_mat_from_yml(const std::vector<std::string>& yml_file_names,
										std::map<std::string, cv::Mat>& mats)
{
	mats.clear();

	for (auto it : yml_file_names)
	{
		cv::Mat img;
		read_grid_img_CV_32FC3(it, img);

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

		create_mat_from_yml(yml_file_names, mats);

		for (auto it : yml_file_names)
		{
			auto it_map = mats.find(it);
			show_direct_grid_img_CV_32FC3(it_map->first, it_map->second);
		}
	}

	return 0;
}
