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
int train_test(void)
{
	CCifar10 cifar10;
	cifar10.set_path("./bin/cifar-10-batches-bin");

	cifar10.load_train_batchs();
	cifar10.load_test_batchs();

	//--------------------------------------------------------------------------
	// Get adequate data from cifar10

	float* train_labels = nullptr;
	unsigned int count_train = cifar10.get_all_train_labels(&train_labels);

	float* test_labels = nullptr;
	unsigned int count_test = cifar10.get_all_test_labels(&test_labels);

	float* train_imgs = nullptr;
	count_train = cifar10.get_all_train_batch_img(&train_imgs);

	float* test_imgs = nullptr;
	count_test = cifar10.get_all_test_batch_img(&test_imgs);

	//--------------------------------------------------------------------------
	// Test RGB image from cifar10
	//cifar10.show_train_img(3, 1500);
	//--------------------------------------------------------------------------

	caffe::Caffe::set_mode(caffe::Caffe::GPU);
	//caffe::Caffe::set_mode(caffe::Caffe::CPU);

	caffe::SolverParameter solver_param;
//    caffe::ReadSolverParamsFromTextFileOrDie("./models/solver.prototxt", &solver_param);
	caffe::ReadSolverParamsFromTextFileOrDie("./models/cifar10_full_solver.prototxt", &solver_param);
	std::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

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
    auto input_blob = solver->net()->blob_by_name("data");

    std::cout << "channels: " << input_blob->channels() << std::endl;
	std::cout << "height: " << input_blob->height() << std::endl;
	std::cout << "width: " << input_blob->width() << std::endl;
	std::cout << "count: " << input_blob->count() << std::endl;

	float loss = 0.0;
	for (unsigned int uiI = 0; uiI < 100000; uiI++)
	{
		loss = solver->net()->ForwardBackward();
		solver->net()->Update();
	}

	std::cout << "loss " << loss << std::endl;

	auto output_blob = solver->net()->blob_by_name("prob");

	std::cout << "channels: " << output_blob->channels() << std::endl;
	std::cout << "height: " << output_blob->height() << std::endl;
	std::cout << "width: " << output_blob->width() << std::endl;
	std::cout << "count: " << output_blob->count() << std::endl;

	std::cout << "values: " << std::endl;

	for (unsigned int uiI = 0; uiI < output_blob->channels(); uiI++)
	{
		std::cout << "value: " << output_blob->data_at(1, uiI, 1, 1) << std::endl;
	}
	std::cout << std::endl;

  return 0;
}
