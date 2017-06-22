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
 * sudoku_solver is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#include <cmath>
#include <ctime>
#include <chrono>
#include <random>
#include <memory>
#include <algorithm>
#include <iostream>
#include <functional>

#include <sys/stat.h>

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

#include "CCifar10.hpp"

////////////////////////////////////////////////////////////////////////////////
template <typename T> void desc_network(caffe::MemoryDataLayer<T>& layer)
{
	std::cout << layer.batch_size() << " " <<
				layer.channels() << " " <<
				layer.height() << " " <<
				layer.width() << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T> void desc_network(caffe::ConvolutionLayer<T>& layer){}

////////////////////////////////////////////////////////////////////////////////
template <typename T> void desc_network(caffe::PoolingLayer<T>& layer){}

////////////////////////////////////////////////////////////////////////////////
template <typename T> void desc_network(caffe::ReLULayer<T>& layer){}

////////////////////////////////////////////////////////////////////////////////
template <typename T> void desc_network(caffe::LRNLayer<T>& layer){}

////////////////////////////////////////////////////////////////////////////////
template <typename T> void desc_network(caffe::InnerProductLayer<T>& layer){}

////////////////////////////////////////////////////////////////////////////////
template <typename T> void desc_network(caffe::SoftmaxLayer<T>& layer){}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
std::map<std::string, std::function<void(caffe::Layer<T>& layer)> > layer_desc_fn =
{
	{"MemoryData", [](caffe::Layer<T>& layer) -> void {desc_network((caffe::MemoryDataLayer<T>&)layer);}},
	{"Convolution", [](caffe::Layer<T>& layer) -> void {desc_network((caffe::ConvolutionLayer<T>&)layer);}},
	{"Pooling", [](caffe::Layer<T>& layer) -> void {desc_network((caffe::PoolingLayer<T>&)layer);}},
	{"ReLU", [](caffe::Layer<T>& layer) -> void {desc_network((caffe::ReLULayer<T>&)layer);}},
	{"LRN", [](caffe::Layer<T>& layer) -> void {desc_network((caffe::LRNLayer<T>&)layer);}},
	{"InnerProduct", [](caffe::Layer<T>& layer) -> void {desc_network((caffe::InnerProductLayer<T>&)layer);}},
	{"Softmax", [](caffe::Layer<T>& layer) -> void {desc_network((caffe::SoftmaxLayer<T>&)layer);}}
};

////////////////////////////////////////////////////////////////////////////////
/**
 *
 * @param net
 */
template <typename T>
void desc_network(caffe::Net<T>& net)
{
	std::cout << "Network name: " << net.name() << std::endl;
	const std::vector<std::string>& vector_blob_names = net.blob_names();

	for (const auto& it_blob_names : vector_blob_names)
	{
		auto boost_ptr_blob = net.blob_by_name(it_blob_names);
		caffe::Blob<T>* blob = boost_ptr_blob.get();
		std::cout << it_blob_names << " -> " <<
					blob->num() << " " <<
					blob->channels() << " " <<
					blob->height() << " " <<
					blob->width() << " " <<
					blob->count() << std::endl;
	}

	const std::vector<std::string>& layer_names = net.layer_names();

	for (const auto& it_layer_names : layer_names)
	{
		auto boost_ptr_layer = net.layer_by_name(it_layer_names);
		caffe::Layer<T>* layer = boost_ptr_layer.get();
		std::cout << it_layer_names << " -> " <<
				layer->ExactNumTopBlobs() << " " <<
				layer->ExactNumBottomBlobs() << " " <<
				layer->type() << std::endl;

		auto it = layer_desc_fn<T>.find(layer->type());
		if (it != layer_desc_fn<T>.end())
		{
			it->second(*layer);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
/**
 *
 * @param file_name
 * @return
 */
bool exist_file(const std::string& file_name)
{
	bool result = false;

	struct stat st;

	if (stat(file_name.c_str(), &st) == 0) result = true;

	return result;
}

////////////////////////////////////////////////////////////////////////////////
/**
 *
 * @param cifar10
 * @return
 */
int train_test(CCifar10* cifar10, struct S_ConfigArgs* configArgs)
{
	//--------------------------------------------------------------------------
	// Get adequate data from cifar10
	float* train_labels = nullptr;
	unsigned int count_train = cifar10->get_all_train_labels(&train_labels);

	float* test_labels = nullptr;
	unsigned int count_test = cifar10->get_all_test_labels(&test_labels);

	float* train_imgs = nullptr;
	count_train = cifar10->get_all_train_batch_img(&train_imgs);

	float* test_imgs = nullptr;
	count_test = cifar10->get_all_test_batch_img(&test_imgs);
	//--------------------------------------------------------------------------

	caffe::Caffe::set_mode(caffe::Caffe::GPU);
	//caffe::Caffe::set_mode(caffe::Caffe::CPU);

	if (!exist_file("cifar10_full_iter_60000.solverstate.h5"))
	{
		caffe::SolverParameter solver_param;
	//    caffe::ReadSolverParamsFromTextFileOrDie("./models/solver.prototxt", &solver_param);
		caffe::ReadSolverParamsFromTextFileOrDie("./models/cifar10_full_solver.prototxt", &solver_param);
		std::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

	//	solver_param.base_lr();
	//	solver_param.set_base_lr(0.001);

		//	caffe::Net<float> net_g("./models/g.prototxt", caffe::Phase::TRAIN);
		//	caffe::Net<float> net_d("./models/d.prototxt", caffe::Phase::TRAIN);

		caffe::MemoryDataLayer<float> *dataLayer_trainnet =
			(caffe::MemoryDataLayer<float> *) (solver->net()->layer_by_name("cifar10").get());
		caffe::MemoryDataLayer<float> *dataLayer_testnet =
			(caffe::MemoryDataLayer<float> *) (solver->test_nets()[0]->layer_by_name("cifar10").get());

	//    dataLayer_trainnet->Reset(train_imgs, train_labels, CCifar10::cifar10_imgs_batch_s);
	//    dataLayer_testnet->Reset(test_imgs, test_labels, CCifar10::cifar10_imgs_batch_s);

		dataLayer_trainnet->Reset(train_imgs, train_labels, count_train);
		dataLayer_testnet->Reset(test_imgs, test_labels, count_test);

		const caffe::LayerParameter param = dataLayer_trainnet->layer_param();
		desc_network(*(solver->net().get()));
		param.PrintDebugString();

		//--------------------------------------------------------------------------
		solver->Solve();
		//--------------------------------------------------------------------------
	}

	if (!exist_file("cifar10_full_iter_65000.solverstate.h5"))
	{
		caffe::SolverParameter solver_param;
		caffe::ReadSolverParamsFromTextFileOrDie("./models/cifar10_full_solver_lr1.prototxt", &solver_param);
		std::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
		solver->Restore("cifar10_full_iter_60000.solverstate.h5");

		caffe::MemoryDataLayer<float> *dataLayer_trainnet =
			(caffe::MemoryDataLayer<float> *) (solver->net()->layer_by_name("cifar10").get());
		caffe::MemoryDataLayer<float> *dataLayer_testnet =
			(caffe::MemoryDataLayer<float> *) (solver->test_nets()[0]->layer_by_name("cifar10").get());

		dataLayer_trainnet->Reset(train_imgs, train_labels, count_train);
		dataLayer_testnet->Reset(test_imgs, test_labels, count_test);

		const caffe::LayerParameter param = dataLayer_trainnet->layer_param();
		desc_network(*(solver->net().get()));
		param.PrintDebugString();

		//--------------------------------------------------------------------------
		solver->Solve();
		//--------------------------------------------------------------------------
	}

	if (!exist_file("cifar10_full_iter_70000.solverstate.h5"))
	{
		caffe::SolverParameter solver_param;
		caffe::ReadSolverParamsFromTextFileOrDie("./models/cifar10_full_solver_lr2.prototxt", &solver_param);
		std::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
		solver->Restore("cifar10_full_iter_65000.solverstate.h5");

		caffe::MemoryDataLayer<float> *dataLayer_trainnet =
				(caffe::MemoryDataLayer<float> *) (solver->net()->layer_by_name("cifar10").get());
		caffe::MemoryDataLayer<float> *dataLayer_testnet =
				(caffe::MemoryDataLayer<float> *) (solver->test_nets()[0]->layer_by_name("cifar10").get());

		dataLayer_trainnet->Reset(train_imgs, train_labels, count_train);
		dataLayer_testnet->Reset(test_imgs, test_labels, count_test);

		const caffe::LayerParameter param = dataLayer_trainnet->layer_param();
		desc_network(*(solver->net().get()));
		param.PrintDebugString();

		//----------------------------------------------------------------------
		solver->Solve();
		//----------------------------------------------------------------------
	}

	caffe::Net<float> net_final("./models/d_1_test.prototxt", caffe::Phase::TEST);
	net_final.CopyTrainedLayersFromHDF5("cifar10_full_iter_70000.caffemodel.h5");

	auto input = net_final.blob_by_name("data");
	input->Reshape({1, 3, 32, 32});
	float *data = input->mutable_cpu_data();
	const int n = input->count();

	unsigned int img_index = 1234;

	memcpy(data, test_imgs + (img_index * 3072) , 3 * 32 * 32 * sizeof(float));

	net_final.Forward();

	auto input_blob = net_final.blob_by_name("ip1");

	std::cout << "channels: " << input_blob->channels() << std::endl;
	std::cout << "height: " << input_blob->height() << std::endl;
	std::cout << "width: " << input_blob->width() << std::endl;
	std::cout << "count: " << input_blob->count() << std::endl;
	std::cout << "num: " << input_blob->num() << std::endl;

	std::cout << "values: " << std::endl;

	unsigned int index = 0;
	for (unsigned int n = 0; n < input_blob->num(); n++)
		for (unsigned int c = 0; c < input_blob->channels(); c++)
			for (unsigned int h = 0; h < input_blob->height(); h++)
				for (unsigned int w = 0; w < input_blob->width(); w++)
				{
					if (index % 20 == 0) std::cout << std::endl;
					std::cout << input_blob->data_at(n, c, h, w) << " ";
					index++;
				}
	std::cout << std::endl;

	auto output_blob = net_final.blob_by_name("prob");

	std::cout << "channels: " << output_blob->channels() << std::endl;
	std::cout << "height: " << output_blob->height() << std::endl;
	std::cout << "width: " << output_blob->width() << std::endl;
	std::cout << "count: " << output_blob->count() << std::endl;
	std::cout << "num: " << output_blob->num() << std::endl;

	std::cout << "values: " << std::endl;

	index = 0;
	for (unsigned int n = 0; n < output_blob->num(); n++)
		for (unsigned int c = 0; c < output_blob->channels(); c++)
			for (unsigned int h = 0; h < output_blob->height(); h++)
				for (unsigned int w = 0; w < output_blob->width(); w++)
				{
					if (index % 20 == 0) std::cout << std::endl;
					std::cout << output_blob->data_at(n, c, h, w) << " ";
					index++;
				}
	std::cout << std::endl;


	//--------------------------------------------------------------------------
	// Test RGB image from cifar10
	CCifar10::print_cifar10_labels();
	cifar10->show_test_img(img_index);
	//--------------------------------------------------------------------------

	return 0;
}
