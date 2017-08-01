/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-  */
/*
 * CLFWFaceDatabase.hpp
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

/** @file CLFWFaceDatabase.hpp
 * @author Juan Maria Gomez Lopez <juanecitorr@gmail.com>
 * @date 31 Jul 2017
 */

#ifndef INCLUDE_CLFWFACEDATABASE_HPP_
#define INCLUDE_CLFWFACEDATABASE_HPP_

#include <vector>
#include <memory>
#include <map>
#include <iostream>
#include <string>
#include <cstring>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cuda.h>
#include <functional>

/**
 *
 */
template <typename DataType>
struct S_LFW_FaceDB_img_rgb
{
	DataType rgb_[12288]; // 3 * 64 * 64
}__attribute__((packed));


template <typename DataType>
struct S_LFW_FaceDB_img
{
	DataType red_channel_[4096];
	DataType green_channel_[4096];
	DataType blue_channel_[4096];
}__attribute__((packed));

////////////////////////////////////////////////////////////////////////////////
/**
 *
 */
class CLFWFaceDatabase
{

	public:

		CLFWFaceDatabase();
		virtual ~CLFWFaceDatabase(){}


		void set_path(const std::string& path) {path_ = path;}

		void load(void);

		unsigned int get_count_imgs(void) const { return count_imgs_;}

		//----------------------------------------------------------------------

		template <typename T>
		unsigned int __attribute__((warn_unused_result))
			get_imgs(T** imgs)
		{
			struct S_LFW_FaceDB_img<T>* struct_imgs = new S_LFW_FaceDB_img<T> [count_imgs_];

			for (unsigned int uiI = 0; uiI < count_imgs_; uiI++)
			{
				for (unsigned int uiJ = 0; uiJ < (64 * 64); uiJ++)
				{
					S_LFW_FaceDB_img<uint8_t>* inter_imgs = this->data_.get();
					struct_imgs[uiI].red_channel_[uiJ] = inter_imgs[uiI].red_channel_[uiJ];
					struct_imgs[uiI].green_channel_[uiJ] = inter_imgs[uiI].green_channel_[uiJ];
					struct_imgs[uiI].blue_channel_[uiJ] = inter_imgs[uiI].blue_channel_[uiJ];
				}
			}

			*imgs = (T*)struct_imgs;
			return count_imgs_;
		}

		template <typename T>
		unsigned int __attribute__((warn_unused_result))
			get_imgs(T** imgs,
						const std::vector<std::function<unsigned int(T**,
											unsigned int)>>& vector_transf)
		{
			unsigned int count = this->get_imgs<T>(imgs);

			for (auto it_fn : vector_transf)
			{
				count = it_fn(imgs, count);
			}

			return count;
		}

		template <typename T>
		unsigned int __attribute__((warn_unused_result))
			get_imgs_rgb(T** imgs);

		template <typename T>
		unsigned int __attribute__((warn_unused_result))
			get_imgs_rgb(T** imgs,
						const std::vector<std::function<unsigned int(T**,
											unsigned int)>>& vector_transf);
		//----------------------------------------------------------------------

		void show_img(uint8_t* img, size_t img_size)
		{
			const int channels = 3;
			const int height = 64;
			const int width = 64;

			// CV_8UC3 -> RGB 0 - 255
			// CV_8SC3 -> RGB -127 - 127
			int type = CV_8UC3;

			cv::Mat canvas2(height, width, type , img);

			cv::imshow("faces", canvas2);
			cv::waitKey();
		}

		void show_img(unsigned int img_index)
		{
			uint8_t* img = (uint8_t*)(this->rgb_data_.get() + img_index);
			show_img(img, 3 * 64 * 64);
		}

	private:

		void calculate_means(void);

		bool is_loaded_;

		unsigned int count_imgs_;

		std::string path_;

		std::shared_ptr<struct S_LFW_FaceDB_img<uint8_t> > data_;
		std::shared_ptr<struct S_LFW_FaceDB_img_rgb<uint8_t> > rgb_data_;
};


#endif /* INCLUDE_CLFWFACEDATABASE_HPP_ */
