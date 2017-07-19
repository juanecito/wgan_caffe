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
#include <iomanip>
#include <vector>

#include <pthread.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <caffe/caffe.hpp>

#include <caffe/layers/memory_data_layer.hpp>
#include <caffe/layers/conv_layer.hpp>
#include <caffe/layers/pooling_layer.hpp>
#include <caffe/layers/relu_layer.hpp>
#include <caffe/layers/lrn_layer.hpp>
#include <caffe/layers/inner_product_layer.hpp>
#include <caffe/layers/softmax_layer.hpp>
#include <caffe/layers/customizable_loss_layer.hpp>

#include <caffe/solver.hpp>
#include <caffe/sgd_solvers.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "CCifar10.hpp"
#include "CCustomLossLayerBackwardGPU.hpp"



//------------------------------------------------------------------------------
// Error in library libcaffe.so temporal fix
template class caffe::SolverRegistry<float>;

////////////////////////////////////////////////////////////////////////////////
void show_img_CV_32FC1(unsigned int img_width, unsigned int img_height, const float* img_data);

////////////////////////////////////////////////////////////////////////////////
void show_grid_img_CV_32FC3(unsigned int img_width, unsigned int img_height,
			const float* img_data, unsigned int channels,
			unsigned int grid_width, unsigned int grid_height);

void write_grid_img_CV_32FC3(const std::string& file_name,
		unsigned int img_width, unsigned int img_height, const float* img_data,
		unsigned int channels, unsigned int grid_width, unsigned int grid_height);

////////////////////////////////////////////////////////////////////////////////
template <typename T>
class CClampFunctor: public caffe::Net<T>::Callback
{
	public:

		CClampFunctor(caffe::Net<T>& net, T clamp_lower,T clamp_upper):
			net_(net), clamp_lower_(clamp_lower), clamp_upper_(clamp_upper){}

	protected:

		virtual void run(int layer)
		{
			std::cout << "layer: " << layer << std::endl;
			std::vector<caffe::Blob<T>*>& learnable_params =
				const_cast<std::vector<caffe::Blob<T>*>& >(net_.learnable_params());
			this->clamp(learnable_params.at(layer));
		}

		virtual ~CClampFunctor(){}

		friend class caffe::Net<T>;

	private:

		void clamp(caffe::Blob<T>* blob)
		{
			T* data = blob->mutable_cpu_data();
			unsigned int count = blob->count();

			for (unsigned int uiI = 0; uiI < count; uiI++)
			{
				if (data[uiI] < clamp_lower_){ data[uiI] = clamp_lower_; }
				else if (data[uiI] > clamp_upper_){ data[uiI] = clamp_upper_; }
			}
		}

		caffe::Net<T>& net_;

		T clamp_lower_;
		T clamp_upper_;
};


////////////////////////////////////////////////////////////////////////////////
void get_data_from_cifar10(CCifar10* cifar10,
		float** train_labels, float** test_labels,
		float** train_imgs, float** test_imgs,
		unsigned int& count_train, unsigned int& count_test);

void initialize_network_weights(caffe::Net<float>* net);

void recalculateZ(float * z_data);

////////////////////////////////////////////////////////////////////////////////
int main_test_2(CCifar10* cifar10_data)
{

	float* z_data = new float [64 * 100];
	float* z_fix_data = new float [64 * 100];

	recalculateZ(z_fix_data);

	int iRC = 0;

	float* train_labels = nullptr;
	float* test_labels = nullptr;
	float* train_imgs = nullptr;
	float* test_imgs = nullptr;
	unsigned int count_train = 0;
	unsigned int count_test = 0;

	get_data_from_cifar10(cifar10_data,
			&train_labels, &test_labels, &train_imgs, &test_imgs,
			count_train, count_test);

	//--------------------------------------------------------------------------

	//caffe::Caffe::set_mode(caffe::Caffe::CPU);
	caffe::Caffe::set_mode(caffe::Caffe::GPU);

	int batch_size = 64;

	float* ones = nullptr;
	float* mones = nullptr;

	ones = new float [batch_size];
	mones = new float [batch_size];

	for (unsigned int uiI = 0; uiI < batch_size; uiI++)
	{
		ones[uiI] = 1.0;
		mones[uiI] = -1.0;
	}


	caffe::SolverParameter solver_param;
	boost::shared_ptr<caffe::Solver<float> > solver;

	caffe::ReadSolverParamsFromTextFileOrDie("./models2/solver_u.prototxt", &solver_param);
	solver.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param));

	caffe::Net<float>* net_u = solver->net().get();


	unsigned int d_iter = 25;
	unsigned int main_it = 1000;

	auto input_g = net_u->blob_by_name("data");
	auto input_d = net_u->blob_by_name("gconv5");

	auto input_label_d = net_u->blob_by_name("label");

	input_g->Reshape({batch_size, 100, 1, 1});
	input_label_d->Reshape({batch_size, 1, 1, 1});

	for (unsigned int uiI = 0; uiI < main_it; uiI++)
	{
		float* data_label = nullptr;
		float* data_d = nullptr;
		for (unsigned int uiJ = 0; uiJ < d_iter; uiJ++)
		{

			//------------------------------------------------------------------
			// Train D with real

			net_u->ClearParamDiffs();

#if 0
			auto learnable_blobs = net_u->learnable_params();

			for (auto it_lb : learnable_blobs)
			{
				std::cout << (void*)it_lb << std::endl;
				std::cout << it_lb->shape_string() << std::endl;
			}
			std::cout << "======================================" << std::endl;
			const std::vector<std::string>& blob_names= net_u->blob_names();
			for (auto it_b_name : blob_names)
			{
				std::cout << it_b_name << " ";
			}
			std::cout << std::endl;
			std::cout << "======================================" << std::endl;

			auto param_display_names = net_u->param_display_names();
			for (auto it_param_names : param_display_names)
			{
				std::cout << "(" << it_param_names<< " " << ")";
			}
			std::cout << std::endl;


			auto param_layer_indices = net_u->param_layer_indices();
			for (auto it_param_layer_indices : param_layer_indices)
			{
				std::cout << "(" << it_param_layer_indices.first << " " << it_param_layer_indices.second << ")";
			}
			std::cout << std::endl;

			auto learnable_param_ids = net_u->learnable_param_ids();
			for (auto it_learnable_param_ids : learnable_param_ids)
			{
				std::cout << "(" << it_learnable_param_ids << " " << ")";
			}
			std::cout << std::endl;

			std::cout << "======================================" << std::endl;


			auto layers = net_u->layers();

			const std::vector<std::vector<caffe::Blob<float>*> >& bottons_by_layer = net_u->bottom_vecs();
			const std::vector<std::vector<caffe::Blob<float>*> >& tops_by_layer = net_u->top_vecs();

			unsigned int layer_idx = 0;
			for (auto it_layer : layers)
			{
				std::cout << "------------------" << std::endl;
				std::cout << it_layer->layer_param().name() << std::endl;

				for (auto it_layer_blobs : it_layer->blobs())
				{
					std::cout << " -> " << it_layer_blobs->shape_string() << " ";
				}
				std::cout << std::endl;

				std::cout << it_layer->layer_param().name() << std::endl;

				for (auto it_botton : bottons_by_layer.at(layer_idx))
				{
					std::cout << (void*)it_botton << " ";
				}
				std::cout << std::endl;
				for (auto it_top : tops_by_layer.at(layer_idx))
				{
					std::cout << (void*)it_top << " ";
				}
				std::cout << std::endl;

				const std::vector<int> & bottom_ids = net_u->bottom_ids(layer_idx);
				for (auto it_bottom_id : bottom_ids)
				{
					std::cout << it_bottom_id << " ";
				}
				std::cout << std::endl;

				const std::vector<int> & top_ids = net_u->top_ids(layer_idx);
				for (auto it_top_id : top_ids)
				{
					std::cout << it_top_id << " ";
				}
				std::cout << std::endl;
				layer_idx++;
			}

#endif

			data_d = input_d->mutable_cpu_data();
			data_label = input_label_d->mutable_cpu_data();

			memcpy(data_d, train_imgs + (uiI * batch_size * 3 * 64 * 64),
					batch_size * 3 * 64 * 64 * sizeof(float));
			memcpy(data_label, ones, batch_size * sizeof(float));

			float loss_D = net_u->ForwardFromTo(19, 32);
			float errorD_real = 0.0;
			if (uiJ == (d_iter - 1))
			{
				unsigned int c = net_u->blob_by_name("Dfc7")->count();
				const float* data_conv5 = net_u->blob_by_name("Dfc7")->cpu_data();

				for (unsigned uiK = 0; uiK < c; uiK++)
				{
					errorD_real += data_conv5[uiK];
				}
				errorD_real /= (float)(c);
			}
			net_u->BackwardFromTo(32, 19);
			solver->ApplyUpdateFromTo(19, 32);
			net_u->ClearParamDiffs();
			//------------------------------------------------------------------
			// Train D with fake

			recalculateZ(z_data);

			float* data_g = input_g->mutable_cpu_data();
			memcpy(data_g, z_data, batch_size * 100 * sizeof(float));

			data_label = input_label_d->mutable_cpu_data();
			memcpy(data_label, mones, batch_size * sizeof(float));

			net_u->Forward();
			net_u->BackwardFromTo(32, 19);
			solver->ApplyUpdateFromTo(19, 32);

			if (uiJ == (d_iter - 1))
			{
				float errorD_fake = 0.0;
				unsigned int c = net_u->blob_by_name("Dfc7")->count();
				const float* data_conv5 = net_u->blob_by_name("Dfc7")->cpu_data();

				for (unsigned uiK = 0; uiK < c; uiK++)
				{
					errorD_fake += data_conv5[uiK];
				}
				errorD_fake /= (float)(c);

				std::cout << "=========================================================" << std::endl;
				std::cout << "net_d->ForwardBackward(): " << loss_D << std::endl;
				std::cout << "iteration D " << uiI << std::endl;
				std::cout << "errorD_real: " << errorD_real << std::endl;
				std::cout << "errorD_fake: " << errorD_fake << std::endl;
				float errorD = errorD_real - errorD_fake;
				std::cout << "errorD: " << errorD << std::endl;
			}
		}

		net_u->ClearParamDiffs();
		recalculateZ(z_data);

		memcpy(input_g->mutable_cpu_data(), z_data, batch_size * 100 * sizeof(float));

		data_label = input_label_d->mutable_cpu_data();
		memcpy(data_label, ones, batch_size * sizeof(float));

		net_u->Forward();
		net_u->Backward();


		// I need update only generator side
		// net_u->learnable_params();

		net_u->Update();
		//solver->ApplyUpdateFromTo(0, 18);
		//solver->Step(1);

		if (uiI > 0 && uiI % 10 == 0)
		{
			memcpy(input_g->mutable_cpu_data(), z_fix_data, batch_size * 100 * sizeof(float));
			net_u->Forward();
			const float* img_g_data = net_u->blob_by_name("gconv5")->cpu_data();
			show_grid_img_CV_32FC3(64, 64, img_g_data, 3, 8, 8);
			std::string file_name = std::string("wgan_grid") + std::to_string(uiI) + std::string(".yml");
			write_grid_img_CV_32FC3(file_name, 64, 64, img_g_data, 3, 8, 8);
		}
	}


	delete[] z_data;
	delete[] z_fix_data;

  return 0;
}
