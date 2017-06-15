/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-  */
/*
 * CCifar10.cpp
 * Copyright (C) 2017 Juan Maria Gomez Lopez <juanecitorr@gmail.com>
 *
 * caffe_network is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * sudoku_solver is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/** @file CCifar10.cpp
 * @author Juan Maria Gomez Lopez <juanecitorr@gmail.com>
 * @date 12 Nov 2017
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <iostream>

#include "CCifar10.hpp"

const std::map<uint8_t, std::string> CCifar10::cifar10_labels =
		{	{1, "airplane"},
			{2, "automobile"},
			{3, "bird"},
			{4, "cat"},
			{5, "deer"},
			{6, "dog"},
			{7, "frog"},
			{8, "horse"},
			{9, "ship"},
			{10, "truck"} };

const std::string CCifar10::train_batch_pattern_name_s = "data_batch_%u.bin";
const std::string CCifar10::test_batch_pattern_name_s = "test_batch.bin";


////////////////////////////////////////////////////////////////////////////////
/**
 *
 */
CCifar10::CCifar10(): is_test_loaded_(false), is_train_loaded_(false)
{
	test_batchs_.clear();
	test_labels_.clear();
	train_batchs_.clear();
	train_labels_.clear();

	ori_train_batchs_.clear();
	ori_test_batchs_.clear();
}

////////////////////////////////////////////////////////////////////////////////
/**
 *
 */
CCifar10::~CCifar10()
{

}

////////////////////////////////////////////////////////////////////////////////
/**
 *
 * @param train_batch_index
 */
void CCifar10::load_train_batch_by_index(unsigned int train_batch_index,
										struct S_Cifar10_label_img* lb_imgs,
										S_Cifar10_img_rgb<uint8_t>* imgs,
										uint8_t* labels)
{
	char str_batch1_name[FILENAME_MAX] = {0};
	snprintf(str_batch1_name, FILENAME_MAX - 1, CCifar10::train_batch_pattern_name_s.c_str(), train_batch_index + 1);
	std::string batch_name = std::string(str_batch1_name);
	std::fstream batch_file;
	batch_file.open(path_ + std::string("/") + batch_name, std::ios::in | std::ios::binary);


	if (batch_file.is_open())
	{
		batch_file.read((char*)lb_imgs, sizeof(struct S_Cifar10_label_img) * CCifar10::cifar10_imgs_batch_s);
	}
	else
	{
		std::cerr << "Error open file " << path_ + std::string("/") + batch_name << std::endl;
	}
	batch_file.close();

	/*
    data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
    The first 1024 entries contain the red channel values, the next 1024 the green,
    and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
	*/
	for (unsigned int uiI = 0; uiI < CCifar10::cifar10_imgs_batch_s; uiI++)
	{
		for (unsigned int uiJ = 0; uiJ < 1024; uiJ++)
		{
			imgs[uiI].rgb_[uiJ * 3] = lb_imgs[uiI]. red_channel_[uiJ];
			imgs[uiI].rgb_[uiJ * 3 + 1] = lb_imgs[uiI]. green_channel_[uiJ];
			imgs[uiI].rgb_[uiJ * 3 + 2] = lb_imgs[uiI]. blue_channel_[uiJ];
		}
		labels[uiI] = lb_imgs[uiI].label_;
	}
}

////////////////////////////////////////////////////////////////////////////////
/**
 *
 * @param pattern
 */
void CCifar10::load_train_batchs(void)
{
	struct S_Cifar10_label_img* lb_imgs = nullptr;
	struct S_Cifar10_img_rgb<uint8_t>* imgs = nullptr;
	uint8_t* labels = nullptr;

	for (unsigned int uiI = 0; uiI < CCifar10::cifar10_train_batch_s; uiI++)
	{
		lb_imgs = new struct S_Cifar10_label_img [CCifar10::cifar10_imgs_batch_s];
		memset(lb_imgs, 0, sizeof(struct S_Cifar10_label_img) * CCifar10::cifar10_imgs_batch_s);

		imgs = new struct S_Cifar10_img_rgb<uint8_t> [CCifar10::cifar10_imgs_batch_s];
		memset(imgs, 0, sizeof(struct S_Cifar10_img_rgb<uint8_t>) * CCifar10::cifar10_imgs_batch_s);

		labels = new uint8_t [10000];
		memset(labels, 0, sizeof(uint8_t) * 10000);

		load_train_batch_by_index(uiI, lb_imgs, imgs, labels);

		std::shared_ptr<struct S_Cifar10_img_rgb<uint8_t> > train_batch_img(imgs,
									[](struct S_Cifar10_img_rgb<uint8_t>* p){delete[] p;});
		std::shared_ptr<struct S_Cifar10_label_img> train_batch_label_img(lb_imgs,
									[](struct S_Cifar10_label_img* p){delete[] p;});
		std::shared_ptr<uint8_t> train_label(labels, [](uint8_t* p){delete[] p;});

		train_batchs_.push_back(train_batch_img);
		ori_train_batchs_.push_back(train_batch_label_img);
		train_labels_.push_back(train_label);
	}
}

////////////////////////////////////////////////////////////////////////////////
/**
 *
 * @param train_batch_index
 */
void CCifar10::load_test_batch_by_index(unsigned int test_batch_index,
										struct S_Cifar10_label_img* lb_imgs,
										S_Cifar10_img_rgb<uint8_t>* imgs,
										uint8_t* labels)
{
	char str_batch1_name[FILENAME_MAX] = {0};
	snprintf(str_batch1_name, FILENAME_MAX - 1, CCifar10::test_batch_pattern_name_s.c_str());
	std::string batch_name = std::string(str_batch1_name);
	std::fstream batch_file;
	batch_file.open(path_ + std::string("/") + batch_name, std::ios::in | std::ios::binary);

	if (batch_file.is_open())
	{
		batch_file.read((char*)lb_imgs, sizeof(struct S_Cifar10_label_img) * CCifar10::cifar10_imgs_batch_s);
	}
	else
	{
		std::cerr << "Error open file " << std::endl;
	}
	batch_file.close();



	/*
    data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
    The first 1024 entries contain the red channel values, the next 1024 the green,
    and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
	*/
	for (unsigned int uiI = 0; uiI < CCifar10::cifar10_imgs_batch_s; uiI++)
	{
		for (unsigned int uiJ = 0; uiJ < 1024; uiJ++)
		{
			imgs[uiI].rgb_[uiJ * 3] = lb_imgs[uiI]. red_channel_[uiJ];
			imgs[uiI].rgb_[uiJ * 3 + 1] = lb_imgs[uiI]. green_channel_[uiJ];
			imgs[uiI].rgb_[uiJ * 3 + 2] = lb_imgs[uiI]. blue_channel_[uiJ];
		}
		labels[uiI] = lb_imgs[uiI].label_;
	}
}

////////////////////////////////////////////////////////////////////////////////
/**
 *
 * @param pattern
 */
void CCifar10::load_test_batchs(void)
{
	//Each image is a vector 32x32x3 of unsigned char data
	struct S_Cifar10_label_img* lb_imgs = nullptr;
	struct S_Cifar10_img_rgb<uint8_t>* imgs = nullptr;
	uint8_t* labels = nullptr;

	for (unsigned int uiI = 0; uiI < CCifar10::cifar10_test_batch_s; uiI++)
	{
		lb_imgs = new struct S_Cifar10_label_img [CCifar10::cifar10_imgs_batch_s];
		memset(lb_imgs, 0, sizeof(struct S_Cifar10_label_img) * CCifar10::cifar10_imgs_batch_s);

		imgs = new struct S_Cifar10_img_rgb<uint8_t> [CCifar10::cifar10_imgs_batch_s];
		memset(imgs, 0, sizeof(struct S_Cifar10_img_rgb<uint8_t>) * CCifar10::cifar10_imgs_batch_s);

		labels = new uint8_t [10000];
		memset(labels, 0, sizeof(uint8_t) * 10000);

		load_test_batch_by_index(uiI, lb_imgs, imgs, labels);

		std::shared_ptr<struct S_Cifar10_img_rgb<uint8_t> > test_batch_img(imgs, [](struct S_Cifar10_img_rgb<uint8_t>* p){delete[] p;});
		std::shared_ptr<struct S_Cifar10_label_img> test_batch_label_img(lb_imgs, [](struct S_Cifar10_label_img* p){delete[] p;});
		std::shared_ptr<uint8_t> test_label(labels, [](uint8_t* p){delete[] p;});

		test_batchs_.push_back(test_batch_img);
		ori_test_batchs_.push_back(test_batch_label_img);
		train_labels_.push_back(test_label);
	}
}

////////////////////////////////////////////////////////////////////////////////
/**
 *
 * @param img
 * @param img_size
 */
void CCifar10::show_img(uint8_t* img, size_t img_size)
{
	const int num = 1;
	const int channels = 3;
	const int height = 32;
	const int width = 32;

	// CV_8UC3 -> RGB

	cv::Mat canvas2(height, width, CV_8UC3 , img);
	cv::Mat canvas3(64, 64, CV_8UC3);
	cv::Size size(64, 64);//the dst image size,e.g.100x100
	cv::resize(canvas2, canvas3, size, 2.0, 2.0, cv::INTER_CUBIC);//resize image

	cv::imshow("cifar10", canvas3);
	cv::waitKey();
}

////////////////////////////////////////////////////////////////////////////////
/**
 *
 * @param batch_index
 * @param img_index
 */
void CCifar10::show_train_img(unsigned int batch_index, unsigned int img_index)
{
	uint8_t* img = this->get_train_img_rgb(batch_index, img_index);
	show_img(img, 3072);
}

////////////////////////////////////////////////////////////////////////////////
/**
 *
 * @param batch_index
 * @param img_index
 */
void CCifar10::show_test_img(unsigned int img_index)
{
	uint8_t* img = this->get_test_img_rgb(img_index);
	show_img(img, 3072);
}
