/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-  */
/*
 * main.cpp
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

/** @file main.cpp
 * @author Juan Maria Gomez Lopez <juanecitorr@gmail.com>
 * @date 02 Jun 2017
 */

#include <cmath>
#include <ctime>
#include <chrono>
#include <random>
#include <memory>
#include <algorithm>
#include <iostream>
#include <functional>

#include <caffe/caffe.hpp>

#include <caffe/layers/memory_data_layer.hpp>
#include <caffe/layers/conv_layer.hpp>
#include <caffe/layers/pooling_layer.hpp>
#include <caffe/layers/relu_layer.hpp>
#include <caffe/layers/lrn_layer.hpp>
#include <caffe/layers/inner_product_layer.hpp>
#include <caffe/layers/softmax_layer.hpp>

#include <caffe/solver.hpp>
#include <caffe/sgd_solvers.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "CCifar10.hpp"
#include "config_args.hpp"


int train_test(CCifar10* cifar10_data, struct S_ConfigArgs* configArgs);

//int main_test_2(CCifar10* cifar10_data, struct S_ConfigArgs* psConfigArgs);

int wgan(CCifar10* cifar10_data, struct S_ConfigArgs* psConfigArgs);

////////////////////////////////////////////////////////////////////////////////
template <typename T>
void verify_img(caffe::Net<T>* net, CCifar10* cifar10, bool is_memory_data_layer)
{
	T* test_rgb_imgs = nullptr;
	unsigned int count_test = cifar10->get_all_test_batch_img_rgb(&test_rgb_imgs);

	T* test_imgs = nullptr;
	count_test = cifar10->get_all_test_batch_img(&test_imgs);


	T* test_labels = nullptr;
	count_test = cifar10->get_all_test_labels(&test_labels);

	S_Cifar10_img_rgb<T>* imgs_rgb_str = (S_Cifar10_img_rgb<T>*)test_rgb_imgs;
	S_Cifar10_img<T>* imgs_str = (S_Cifar10_img<T>*)test_imgs;

	unsigned int img_index = 3500;
	if (is_memory_data_layer)
	{
	    caffe::MemoryDataLayer<T> *dataLayer_trainnet =
	    	(caffe::MemoryDataLayer<T> *) (net->layer_by_name("cifar10").get());
		dataLayer_trainnet->Reset((T*)(imgs_str + img_index), (T*)(test_labels + img_index), 100);
		std::cout << "label: " << test_labels[img_index] << std::endl;
	}
	else
	{
		const boost::shared_ptr<caffe::Blob<T> > sp_input = net->blob_by_name("data");

		caffe::Blob<T>* input = sp_input.get();
		input->Reshape({1, 3, 32, 32});
		T *data = input->mutable_cpu_data();
		const int n = input->count();

		std::cout << "n: " << n << std::endl;

		auto it = CCifar10::cifar10_labels.find(test_labels[img_index]);
		std::cout << "label: " << it->second << std::endl;

		memcpy(data, &(imgs_str[img_index]), 3072 * sizeof(T));
	}

	cifar10->show_test_img(img_index);

	net->Forward();

	const std::vector<caffe::Blob<T>* > output_blobs_vector = net->output_blobs();

	std::cout << "output blobs number " << output_blobs_vector.size() << std::endl;

	for (const auto &it : output_blobs_vector)
	{
		std::cout << it->shape_string() << std::endl;
		unsigned int blob_count = it->count();
		const T* data_result = it->cpu_data();
		for (unsigned int uiI = 0; uiI < blob_count; uiI++)
		{
			if (uiI % 20 == 0) std::cout << std::endl;
			std::cout << " ["<< data_result[uiI] << "] ";
		}
	}

	std::cout << std::endl;

	const boost::shared_ptr<caffe::Blob<T> > sh_ptr_blob = net->blob_by_name("ip1");
	//const boost::shared_ptr<caffe::Blob<T> > sh_ptr_blob = net->blob_by_name("prob");
	caffe::Blob<T>* blob = sh_ptr_blob.get();

	const T* data_result = blob->cpu_data();
	unsigned int blob_count = blob->count();
	unsigned int blob_num = blob->num();
	std::string shape_string = blob->shape_string();

	std::cout << "blob_count: " << blob_count << std::endl;
	std::cout << "blob_num: " << blob_num << std::endl;
	std::cout << "shape_string: " << blob->shape_string() << std::endl;

	for (unsigned int uiI = 0; uiI < blob_count; uiI++)
	{
		if (uiI > 0 && uiI % 20 == 0) std::cout << std::endl;
		std::cout << data_result[uiI] << " ";
	}
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
void verify_img(caffe::Solver<T>* solver, CCifar10* cifar10, bool is_memory_data_layer)
{
	T* test_rgb_imgs = nullptr;
	unsigned int count_test = cifar10->get_all_test_batch_img_rgb(&test_rgb_imgs);

	T* test_imgs = nullptr;
	count_test = cifar10->get_all_test_batch_img(&test_imgs);


	T* test_labels = nullptr;
	count_test = cifar10->get_all_test_labels(&test_labels);

	S_Cifar10_img_rgb<T>* imgs_rgb_str = (S_Cifar10_img_rgb<T>*)test_rgb_imgs;
	S_Cifar10_img<T>* imgs_str = (S_Cifar10_img<T>*)test_imgs;

	unsigned int img_index = 3500;
	if (is_memory_data_layer)
	{
	    caffe::MemoryDataLayer<T> *dataLayer_trainnet =
	    	(caffe::MemoryDataLayer<T> *) (solver->net()->layer_by_name("cifar10").get());
		dataLayer_trainnet->Reset((T*)(imgs_str + img_index), (T*)(test_labels + img_index), 100);
		std::cout << "label: " << test_labels[img_index] << std::endl;
	}
	else
	{
		const boost::shared_ptr<caffe::Blob<T> > sp_input = solver->net()->blob_by_name("data");

		caffe::Blob<T>* input = sp_input.get();
		input->Reshape({1, 3, 32, 32});
		T *data = input->mutable_cpu_data();
		const int n = input->count();

		std::cout << "n: " << n << std::endl;

		auto it = CCifar10::cifar10_labels.find(test_labels[img_index]);
		std::cout << "label: " << it->second << std::endl;

		memcpy(data, &(imgs_str[img_index]), 3072 * sizeof(T));
	}

	cifar10->show_test_img(img_index);

	solver->net()->Forward();

	const std::vector<caffe::Blob<T>* > output_blobs_vector = solver->net()->output_blobs();

	std::cout << "output blobs number " << output_blobs_vector.size() << std::endl;

	for (const auto &it : output_blobs_vector)
	{
		std::cout << it->shape_string() << std::endl;
		unsigned int blob_count = it->count();
		const T* data_result = it->cpu_data();
		for (unsigned int uiI = 0; uiI < blob_count; uiI++)
		{
			if (uiI % 20 == 0) std::cout << std::endl;
			std::cout << " ["<< data_result[uiI] << "] ";
		}
	}

	std::cout << std::endl;

	const boost::shared_ptr<caffe::Blob<T> > sh_ptr_blob = solver->net()->blob_by_name("ip1");
	//const boost::shared_ptr<caffe::Blob<T> > sh_ptr_blob = net->blob_by_name("prob");
	caffe::Blob<T>* blob = sh_ptr_blob.get();

	const T* data_result = blob->cpu_data();
	unsigned int blob_count = blob->count();
	unsigned int blob_num = blob->num();
	std::string shape_string = blob->shape_string();

	std::cout << "blob_count: " << blob_count << std::endl;
	std::cout << "blob_num: " << blob_num << std::endl;
	std::cout << "shape_string: " << blob->shape_string() << std::endl;

	for (unsigned int uiI = 0; uiI < blob_count; uiI++)
	{
		if (uiI > 0 && uiI % 20 == 0) std::cout << std::endl;
		std::cout << data_result[uiI] << " ";
	}
}

////////////////////////////////////////////////////////////////////////////////
/**
 *
 * @param cifar_path
 * @return
 */
CCifar10* get_cifar10_data(const std::string& cifar_path)
{
	CCifar10* cifar10 = new CCifar10();
	cifar10->set_path(cifar_path);

	cifar10->load_train_batchs();
	cifar10->load_test_batchs();

	return cifar10;
}

////////////////////////////////////////////////////////////////////////////////
/**
 *
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
	struct S_ConfigArgs configArgs;

	parse_arguments(&configArgs, argc, argv);

	// Check arguments, set defaults values
	if (configArgs.batch_size_ == 0) configArgs.batch_size_ = 64;
	if (configArgs.z_vector_size_ == 0)configArgs.z_vector_size_ = 100;
	if (configArgs.d_iters_by_g_iter_ == 0) configArgs.d_iters_by_g_iter_ = 25;
	if (configArgs.main_iters_ == 0)configArgs.main_iters_ = 100;
	if (configArgs.solver_d_model_.size() == 0)
	{
		configArgs.solver_d_model_ = "./models/solver_d_lr_A.prototxt";
	}
	if (configArgs.solver_g_model_.size() == 0)
	{
		configArgs.solver_g_model_ = "./models/solver_g_lr_A.prototxt";
	}
	if (configArgs.data_source_folder_path_.size() == 0)
	{
		configArgs.data_source_folder_path_ = "./bin/cifar-10-batches-bin";
	}

	std::cout << "Arguments: " << std::endl;
	std::cout << "Log file: " << configArgs.logarg_ << std::endl;
	std::cout << "solver_d_model_: " << configArgs.solver_d_model_ << std::endl;
	std::cout << "solver_g_model_: " << configArgs.solver_g_model_ << std::endl;
	std::cout << "solver_d_state_: " << configArgs.solver_d_state_ << std::endl;
	std::cout << "solver_g_state_: " << configArgs.solver_g_state_ << std::endl;
	std::cout << "data_source_folder_path: " << configArgs.data_source_folder_path_ << std::endl;
	std::cout << "run wgan: " << configArgs.run_wgan_ << std::endl;
	std::cout << "run cifar10 training: " << configArgs.run_cifar10_training_ << std::endl;
	std::cout << "run cifar10 test: " << configArgs.test_cifar10_ << std::endl;
	std::cout << "batch size: " << configArgs.batch_size_ << std::endl;
	std::cout << "z_vector_bin_file: " << configArgs.z_vector_bin_file_ << std::endl;
	std::cout << "z_vector_size: " << configArgs.z_vector_size_ << std::endl;
	std::cout << "d_iters_by_g_iter: " << configArgs.d_iters_by_g_iter_ << std::endl;
	std::cout << "main_iters: " << configArgs.main_iters_ << std::endl;

	CCifar10* cifar10_data = get_cifar10_data(configArgs.data_source_folder_path_);
	std::unique_ptr<CCifar10> cifar10_sh(cifar10_data);

	int iRC = 0;

	if (configArgs.run_wgan_)
	{
		iRC |= wgan(cifar10_data, &configArgs);
	}

	if (configArgs.run_cifar10_training_ || configArgs.test_cifar10_)
	{
		iRC |= train_test(cifar10_data, &configArgs);
	}

	return iRC;

}
