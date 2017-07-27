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
#include <fstream>

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


#include "config_args.hpp"


static pthread_mutex_t solvers_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t solvers_cond = PTHREAD_COND_INITIALIZER;
static pthread_barrier_t solvers_barrier;

#define OWNER_D 0
#define OWNER_G 1

#define RUN_GPU 1

static int current_owner;


static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


//------------------------------------------------------------------------------
// Error in library libcaffe.so temporal fix
template class caffe::SolverRegistry<float>;

////////////////////////////////////////////////////////////////////////////////
void takeToken(int owner)
{
	/*
	   Lock mutex and wait for signal. Note that the pthread_cond_wait
	   routine will automatically and atomically unlock mutex while it waits.
	 */
	pthread_mutex_lock(&solvers_mutex);
	while (owner != current_owner)
	{
		pthread_cond_wait(&solvers_cond, &solvers_mutex);
		//printf("watch_count(): thread %d Condition signal received.\n", owner);
	}

	//printf("take token: thread %d.\n", owner);
	pthread_mutex_unlock(&solvers_mutex);

}

////////////////////////////////////////////////////////////////////////////////
void releaseToken(int owner)
{
	pthread_mutex_lock(&solvers_mutex);

	/*
    Check the value of count and signal waiting thread when condition is
    reached.  Note that this occurs while mutex is locked.
	 */
	if (owner == current_owner)
	{
		current_owner += 1; current_owner %= 2;
		pthread_cond_signal(&solvers_cond);
		//printf("start: thread %d.\n", owner);
	}
	//printf("release token thread %d  unlocking mutex\n", owner);
	pthread_mutex_unlock(&solvers_mutex);
}

////////////////////////////////////////////////////////////////////////////////
void forward_cpu_g_loss(const std::vector<caffe::Blob<float>*>&,
  	  	  const std::vector<caffe::Blob<float>*>&)
{
	std::cout << "forward_cpu_g_loss " << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
void forward_gpu_g_loss(const std::vector<caffe::Blob<float>*>&,
  	  	  const std::vector<caffe::Blob<float>*>&)
{
	std::cout << "forward_gpu_g_loss " << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
void backward_cpu_g_loss(const std::vector<caffe::Blob<float>*>&,
						const std::vector<bool>&,
						const std::vector<caffe::Blob<float>*>&)
{
	std::cout << "backward_cpu_g_loss " << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
void backward_gpu_g_loss(const std::vector<caffe::Blob<float>*>&,
						const std::vector<bool>&,
						const std::vector<caffe::Blob<float>*>&)
{
	std::cout << "backward_gpu_g_loss " << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
void show_img_CV_32FC1(unsigned int img_width, unsigned int img_height, const float* img_data)
{
//	Test images
	const cv::Mat img(img_width, img_height, CV_32FC1, (void*)img_data);
	cv::imshow("cifar10", img);
	cv::waitKey();
}

////////////////////////////////////////////////////////////////////////////////
void show_grid_img_CV_32FC3(unsigned int img_width, unsigned int img_height, const float* img_data, unsigned int channels,
			unsigned int grid_width, unsigned int grid_height)
{
	unsigned int img_count = grid_height * grid_width;
	unsigned int img_size_per_channel = img_height * img_width;
	unsigned int img_size = channels * img_size_per_channel;

	unsigned int grid_img_count = img_count * img_size;

	float* tranf_img_data = new float [grid_img_count];

	for (unsigned int y_grid = 0; y_grid < grid_height; y_grid++)
	{
		for (unsigned int x_grid = 0; x_grid < grid_width; x_grid++)
		{
			for (unsigned int y_img = 0; y_img < img_height; y_img++)
			{
				for (unsigned int x_img = 0; x_img < img_width; x_img++)
				{
					unsigned int tranf_img_data_index =
								y_grid * grid_width * img_size +
								y_img * grid_width * img_width * channels +
								x_grid * (img_width * channels) +
								x_img * channels;

					unsigned int img_data_index = (y_grid * grid_width + x_grid) * img_size
								+ y_img * img_width + x_img;

					for (unsigned int c = 0; c < channels; c++)
					{
						tranf_img_data[tranf_img_data_index + c] = img_data[img_data_index + c * img_size_per_channel];
					}

				}
			}
		}
	}



	const cv::Mat img(img_width * grid_width, img_height * grid_height, CV_32FC3, tranf_img_data);
	cv::imshow("cifar10_generator", img);
	cv::waitKey();

	delete[] tranf_img_data;
}

////////////////////////////////////////////////////////////////////////////////
void write_grid_img_CV_32FC3(const std::string& file_name,
		unsigned int img_width, unsigned int img_height, const float* img_data,
		unsigned int channels, unsigned int grid_width, unsigned int grid_height)
{
	unsigned int img_count = grid_height * grid_width;
	unsigned int img_size_per_channel = img_height * img_width;
	unsigned int img_size = channels * img_size_per_channel;

	unsigned int grid_img_count = img_count * img_size;

	float* tranf_img_data = new float [grid_img_count];

	for (unsigned int y_grid = 0; y_grid < grid_height; y_grid++)
	{
		for (unsigned int x_grid = 0; x_grid < grid_width; x_grid++)
		{
			for (unsigned int y_img = 0; y_img < img_height; y_img++)
			{
				for (unsigned int x_img = 0; x_img < img_width; x_img++)
				{
					unsigned int tranf_img_data_index =
								y_grid * grid_width * img_size +
								y_img * grid_width * img_width * channels +
								x_grid * (img_width * channels) +
								x_img * channels;

					unsigned int img_data_index = (y_grid * grid_width + x_grid) * img_size
								+ y_img * img_width + x_img;

					for (unsigned int c = 0; c < channels; c++)
					{
						tranf_img_data[tranf_img_data_index + c] = img_data[img_data_index + c * img_size_per_channel];
					}

				}
			}
		}
	}

	const cv::Mat img(img_width * grid_width, img_height * grid_height, CV_32FC3, tranf_img_data);

	cv::FileStorage fs(file_name.c_str(), cv::FileStorage::WRITE);
	fs << "grid_img" << img;
	fs.release();

	delete[] tranf_img_data;
}


////////////////////////////////////////////////////////////////////////////////
unsigned int scale(unsigned int batch_count, unsigned int channels,
			unsigned int width, unsigned int height, float** data,
			unsigned int final_width, unsigned int final_height)
{
	unsigned int data_count = batch_count * channels * width * height;
	unsigned int new_data_count = batch_count * channels * final_width * final_height;
	float* tranf_data = new float[new_data_count];
	memset(tranf_data, 0, new_data_count * sizeof(float));

	cv::Size size(final_width, final_height);

	unsigned int size_img_by_channel = width * height;
	unsigned int size_img = width * height * channels;

	unsigned int final_size_img_by_channel = final_width * final_height;
	unsigned int final_size_img = final_width * final_height * channels;

	for (unsigned int uiI = 0; uiI < batch_count; uiI++)
	{
		for (unsigned int c = 0; c < channels; c++)
		{
			cv::Mat img_ori(height, width, CV_32FC1,
				*data + (uiI * size_img) + c * size_img_by_channel);
			cv::Mat img_final(final_height, final_width, CV_32FC1,
				tranf_data + (uiI * final_size_img) + c * final_size_img_by_channel);
			cv::resize(img_ori, img_final, size, 2.0, 2.0, CV_INTER_LINEAR);//resize image
		}
	}
	delete[] *data;
	*data = tranf_data;

	return batch_count;
}

////////////////////////////////////////////////////////////////////////////////
unsigned int norm(unsigned int batch_count, unsigned int channels,
			unsigned int width, unsigned int height, float** data)
{
	unsigned int data_count = batch_count * channels * width * height;

	for (unsigned int uiI = 0; uiI < data_count; uiI++)
	{
		(*data)[uiI] /= 255.0;
	}

	return batch_count;
}

////////////////////////////////////////////////////////////////////////////////
/*! \brief Timer */
class Timer {
		using Clock = std::chrono::high_resolution_clock;
	public:
		/*! \brief start or restart timer */
		inline void Tic() {
			start_ = Clock::now();
		}
		/*! \brief stop timer */
		inline void Toc() {
			end_ = Clock::now();
		}
		/*! \brief return time in ms */
		inline double Elasped() {
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
			return duration.count();
		}

	private:
		Clock::time_point start_, end_;
};

////////////////////////////////////////////////////////////////////////////////
struct S_InterSolverData
{
	CCifar10* cifar10_;
	float* z_data_;
	float* z_fix_data_;

	unsigned int batch_size_;

	unsigned int d_iter_;
	unsigned int main_it_;

	caffe::Net<float>* net_d_;
	caffe::Net<float>* net_g_;

	float* gpu_ones_;
	float* gpu_zeros_;

	std::string solver_state_file_d_;
	std::string solver_state_file_g_;

	std::string solver_model_file_d_;
	std::string solver_model_file_g_;
};

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
			//std::cout << "layer: " << layer << std::endl;
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
		unsigned int& count_train, unsigned int& count_test)
{

	auto fn_norm = [](float** data, unsigned int count) -> unsigned int {
		return norm(count, 3, 32, 32, data);
		};

	auto fn_scale_64 = [](float** data, unsigned int count) -> unsigned int {
		return scale(count, 3, 32, 32, data, 64, 64);
		};

	pthread_mutex_lock(&solvers_mutex);
	*train_labels = nullptr;
	count_train = cifar10->get_all_train_labels(train_labels);

	*test_labels = nullptr;
	count_test = cifar10->get_all_test_labels(test_labels);

	*train_imgs = nullptr;
	count_train = cifar10->get_all_train_batch_img(train_imgs, {fn_norm, fn_scale_64});

	*test_imgs = nullptr;
	count_test = cifar10->get_all_test_batch_img(test_imgs, {fn_norm, fn_scale_64});
	pthread_mutex_unlock(&solvers_mutex);
}

////////////////////////////////////////////////////////////////////////////////
void initialize_network_weights(caffe::Net<float>* net)
{
	const std::vector<std::string>& layer_names = net->layer_names();
	std::vector<caffe::Blob<float>*>& learnable_params = const_cast<std::vector<caffe::Blob<float>*>&>(net->learnable_params());

	srand(time(NULL));
	std::random_device rd;
	std::mt19937 gen(rd());

	std::normal_distribution<float> nd_conv(0.0, 0.001);
	std::normal_distribution<float> nd_norm(1.0, 0.02);
	std::normal_distribution<float> nd(0.0, 0.1);

//	const int n = input->count();
//	for (int i = 0; i < n; ++i) {
//		data[i] = nd(gen);
//	}

	unsigned int layer_index = 0;

	for (auto it_layer_name : layer_names)
	{
		std::cout << "Layer: " << it_layer_name << std::endl;
		caffe::Blob<float>* blob = learnable_params.at(layer_index);
		const int n = blob->count();
		float* data = blob->mutable_cpu_data();

		auto layer = const_cast<caffe::Layer<float>* >(net->layer_by_name(it_layer_name).get());
		if (it_layer_name.substr(0, 4).compare("conv") == 0)
		{
			for (unsigned int i = 0; i < n; ++i) data[i] = nd_conv(gen);
		}
		else if (it_layer_name.substr(0, 4).compare("norm") == 0)
		{
			for (unsigned int i = 0; i < n; ++i) data[i] = nd_norm(gen);
		}
		else
		{
			for (unsigned int i = 0; i < n; ++i) data[i] = nd(gen);
		}
		layer_index++;
	}
}

////////////////////////////////////////////////////////////////////////////////
template<typename T>
void show_blobs(caffe::Net<T>* net)
{
	const std::vector<std::string>& blobs_names = net->blob_names();

	for (auto it : blobs_names)
	{
		std::cout << "=========================================================" << std::endl;
		std::cout << it << std::endl;
		auto blob = net->blob_by_name(it);

		unsigned int count = blob->count();
		const float* data = blob->cpu_data();
		for (unsigned int uiI = 0; uiI < count; uiI++)
		{
			if (uiI % 100 == 0) std::cout << std::endl;
			std::cout << std::setprecision(10) << data[uiI] << " ";
		}
		std::cout << std::endl << "=========================================================" << std::endl;
		std::cout << std::endl << std::endl;
	}
}

////////////////////////////////////////////////////////////////////////////////
template<typename T>
void show_learneable_params(caffe::Net<T>* net)
{
	const std::vector<caffe::Blob<T>*>& learnable_params = net->learnable_params();

	for (auto blob : learnable_params)
	{
		unsigned int count = blob->count();
		const float* data = blob->cpu_data();
		std::cout << "=========================================================" << std::endl;
		for (unsigned int uiI = 0; uiI < count; uiI++)
		{
			if (uiI % 100 == 0) std::cout << std::endl;
			std::cout << std::setprecision(10) << data[uiI] << " ";
		}
		std::cout << std::endl << "=========================================================" << std::endl;
		std::cout << std::endl << std::endl;
	}
}

////////////////////////////////////////////////////////////////////////////////
template<typename T>
void show_outputs_blobs(caffe::Net<T>* net)
{
	const std::vector<caffe::Blob<float>*>& output_blobs = net->output_blobs();

	for (auto it_blob : output_blobs)
	{
		unsigned int count = it_blob->count();
		const float* data = it_blob->cpu_data();
		std::cout << "=========================================================" << std::endl;
		for (unsigned int uiI = 0; uiI < count; uiI++)
		{
			if (uiI % 100 == 0) std::cout << std::endl;
			std::cout << std::setprecision(10) << data[uiI] << " ";
		}

	}
	std::cout << std::endl << "=========================================================" << std::endl;
	std::cout << std::endl << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
void recalculateZ(float * z_data)
{
	srand(time(NULL));
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> nd(0, 1);

	unsigned int n = 64 * 100 * 1 * 1;
	for (int i = 0; i < n; ++i)
	{
		z_data[i] = nd(gen);
	}
}

////////////////////////////////////////////////////////////////////////////////
void* d_thread_fun(void* interSolverData)
{
	S_InterSolverData* ps_interSolverData = (S_InterSolverData*)interSolverData;

	float* train_labels = nullptr;
	float* test_labels = nullptr;
	float* train_imgs = nullptr;
	float* test_imgs = nullptr;
	unsigned int count_train = 0;
	unsigned int count_test = 0;

	get_data_from_cifar10(ps_interSolverData->cifar10_,
			&train_labels, &test_labels, &train_imgs, &test_imgs,
			count_train, count_test);

	//--------------------------------------------------------------------------

#if defined(RUN_GPU)
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
#else
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
#endif

	caffe::SolverParameter solver_param;
	boost::shared_ptr<caffe::Solver<float> > solver;

	caffe::ReadSolverParamsFromTextFileOrDie(ps_interSolverData->solver_model_file_d_, &solver_param);
	solver.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param));

	int iter_d = 0;

	if (ps_interSolverData->solver_state_file_d_.size() > 0)
	{
		solver->Restore(ps_interSolverData->solver_state_file_d_.c_str());
		iter_d = solver->iter();
	}

	caffe::Net<float>* net_d = ps_interSolverData->net_d_ = solver->net().get();

	pthread_barrier_wait(&solvers_barrier);

	//--------------------------------------------------------------------------
	//initialize_network_weights(net_d.get());
	//CClampFunctor<float>* clampFunctor = new CClampFunctor<float>(*net_d, -0.01, 0.01);
	//net_d->add_before_forward(clampFunctor);

	auto input = net_d->blob_by_name("data");
	auto input_label = net_d->blob_by_name("label");
	input->Reshape({(int)ps_interSolverData->batch_size_, 3, 64, 64});
	input_label->Reshape({(int)ps_interSolverData->batch_size_, 1, 1, 1});

	auto input_g = ps_interSolverData->net_g_->blob_by_name("data");
	input_g->Reshape({64, 100, 1, 1});

	std::fstream output("output.txt", std::ios_base::out);

	for (unsigned int uiI = (iter_d / ps_interSolverData->d_iter_);
			uiI < ps_interSolverData->main_it_ + (iter_d / ps_interSolverData->d_iter_); uiI++)
	{

		// Discriminator and generator threads synchronization
		takeToken(OWNER_D);
		for (unsigned int uiJ = 0; uiJ < ps_interSolverData->d_iter_; uiJ++)
		{
			//------------------------------------------------------------------
			// Train D with real

			float* data_d = input->mutable_cpu_data();
			memcpy(data_d, train_imgs + (uiI * ps_interSolverData->batch_size_ * 3 * 64 * 64),
					ps_interSolverData->batch_size_ * 3 * 64 * 64 * sizeof(float));

			cudaMemcpy(input_label->mutable_gpu_data(), ps_interSolverData->gpu_ones_, ps_interSolverData->batch_size_ * sizeof(float), cudaMemcpyDeviceToDevice);

			float errorD_real = net_d->ForwardBackward();

			//------------------------------------------------------------------
			// Train D with fake
			recalculateZ(ps_interSolverData->z_data_);

			float* data_g = input_g->mutable_cpu_data();
			memcpy(data_g, ps_interSolverData->z_data_, ps_interSolverData->batch_size_ * 100 * sizeof(float));
			ps_interSolverData->net_g_->Forward();
			auto blob_output_g = ps_interSolverData->net_g_->blob_by_name("gconv5");

			cudaMemcpy(input->mutable_gpu_data(), blob_output_g->gpu_data(), ps_interSolverData->batch_size_ * 3 * 64 * 64 * sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(input_label->mutable_gpu_data(), ps_interSolverData->gpu_zeros_, ps_interSolverData->batch_size_ * sizeof(float), cudaMemcpyDeviceToDevice);

			//solver->Step(1);
			solver->StepOne_ForBackAndUpdate();

			if (uiJ == (ps_interSolverData->d_iter_ - 1))
			{
				float errorD_fake = 0.0;
				errorD_fake = net_d->blob_by_name("loss")->cpu_data()[0];

				std::cout << "=========================================================" << std::endl;
				std::cout << "iteration D " << uiI << std::endl;
				std::cout << "errorD_real: " << errorD_real << std::endl;
				std::cout << "errorD_fake: " << errorD_fake << std::endl;
				float errorD = errorD_real + errorD_fake;
				std::cout << "errorD: " << errorD << std::endl;

				output << "=========================================================" << std::endl;
				output << "iteration D " << uiI << std::endl;
				output << "errorD_real: " << errorD_real << std::endl;
				output << "errorD_fake: " << errorD_fake << std::endl;
				output << "errorD: " << errorD << std::endl;
				output.flush();
			}
			net_d->ClearParamDiffs();
		}

		if (uiI > 0 && uiI % 10 == 0)
		{
			solver->Snapshot();
		}

		releaseToken(OWNER_D);
	}

	output.close();

	//--------------------------------------------------------------------------
	pthread_barrier_wait(&solvers_barrier);

	return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
void* g_thread_fun(void* interSolverData)
{
	S_InterSolverData* ps_interSolverData = (S_InterSolverData*)interSolverData;

	float* train_labels = nullptr;
	float* test_labels = nullptr;
	float* train_imgs = nullptr;
	float* test_imgs = nullptr;
	unsigned int count_train = 0;
	unsigned int count_test = 0;

	get_data_from_cifar10(ps_interSolverData->cifar10_,
			&train_labels, &test_labels, &train_imgs, &test_imgs,
			count_train, count_test);

	//--------------------------------------------------------------------------

#if defined(RUN_GPU)
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
#else
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
#endif


	caffe::SolverParameter solver_param;
	caffe::ReadSolverParamsFromTextFileOrDie(ps_interSolverData->solver_model_file_g_, &solver_param);
	std::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

	int iter_g = 0;
	if (ps_interSolverData->solver_state_file_g_.size() > 0)
	{
		solver->Restore(ps_interSolverData->solver_state_file_g_.c_str());
		iter_g = solver->iter();
	}

	caffe::Net<float>* net_g = ps_interSolverData->net_g_ = solver->net().get();


	pthread_barrier_wait(&solvers_barrier);
	//--------------------------------------------------------------------------

	auto input_g = ps_interSolverData->net_g_->blob_by_name("data");
	input_g->Reshape({(int)ps_interSolverData->batch_size_, 100, 1, 1});

	auto blob_output_g = net_g->blob_by_name("gconv5");
	auto net_d_blob_data = ps_interSolverData->net_d_->blob_by_name("data");
	//auto blob_output_d = ps_interSolverData->net_d_->blob_by_name("conv5");

	auto input_label_d = ps_interSolverData->net_d_->blob_by_name("label");

	for (unsigned int uiI = iter_g; uiI < ps_interSolverData->main_it_ + iter_g; uiI++)
	{
		takeToken(OWNER_G);

		recalculateZ(ps_interSolverData->z_data_);

		memcpy(input_g->mutable_cpu_data(), ps_interSolverData->z_data_, ps_interSolverData->batch_size_ * 100 * sizeof(float));
		net_g->Forward();

		//----------------------------------------------------------------------
		// Get Fake

		cudaMemcpy(net_d_blob_data->mutable_gpu_data(), blob_output_g->gpu_data(), ps_interSolverData->batch_size_ * 3 * 64 * 64 * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(input_label_d->mutable_gpu_data(), ps_interSolverData->gpu_ones_, ps_interSolverData->batch_size_ * sizeof(float), cudaMemcpyDeviceToDevice);

		float loss_G = ps_interSolverData->net_d_->ForwardBackward();

		cudaMemcpy(blob_output_g->mutable_gpu_diff(), net_d_blob_data->gpu_diff(), ps_interSolverData->batch_size_ * 3 * 64 * 64 * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(blob_output_g->mutable_gpu_data(), net_d_blob_data->gpu_data(), ps_interSolverData->batch_size_ * 3 * 64 * 64 * sizeof(float), cudaMemcpyDeviceToDevice);

		solver->StepOne_BackAndUpdate();
		//solver->Step(1);

		if (uiI > 0 && uiI % 10 == 0)
		{
			std::cout << "loss_G: " << loss_G << std::endl;
			memcpy(input_g->mutable_cpu_data(), ps_interSolverData->z_fix_data_, ps_interSolverData->batch_size_ * 100 * sizeof(float));
			net_g->Forward();
			const float* img_g_data = net_g->blob_by_name("gconv5")->cpu_data();
			//show_grid_img_CV_32FC3(64, 64, img_g_data, 3, 8, 8);
			std::string file_name = std::string("wgan_grid") + std::to_string(uiI) + std::string(".yml");
			write_grid_img_CV_32FC3(file_name, 64, 64, img_g_data, 3, 8, 8);

			solver->Snapshot();
		}

		net_g->ClearParamDiffs();
		ps_interSolverData->net_d_->ClearParamDiffs();
		releaseToken(OWNER_G);
	}

	pthread_barrier_wait(&solvers_barrier);
	return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
void generate_cuda_data(S_InterSolverData* s_interSolverData)
{
	float* ones = new float [s_interSolverData->batch_size_];
	float* zeros = new float [s_interSolverData->batch_size_];

	for (unsigned int uiI = 0; uiI < s_interSolverData->batch_size_; uiI++)
	{
		ones[uiI] = 1.0;
		zeros[uiI] = 0.0;
	}

	float* ones_gpu = nullptr;
	float* mones_gpu = nullptr;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&(s_interSolverData->gpu_ones_),
								s_interSolverData->batch_size_ * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&(s_interSolverData->gpu_zeros_),
								s_interSolverData->batch_size_ * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(s_interSolverData->gpu_ones_, ones,
			s_interSolverData->batch_size_ * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(s_interSolverData->gpu_zeros_, zeros,
			s_interSolverData->batch_size_ * sizeof(float), cudaMemcpyHostToDevice));

	delete[] ones;
	delete[] zeros;
}

////////////////////////////////////////////////////////////////////////////////
int main_test(CCifar10* cifar10_data, struct S_ConfigArgs* psConfigArgs)
{
	pthread_t thread_d = 0;
	pthread_t thread_g = 0;

	//--------------------------------------------------------------------------
	struct S_InterSolverData s_interSolverData;
	s_interSolverData.cifar10_ = cifar10_data;

	s_interSolverData.net_d_ = nullptr;
	s_interSolverData.net_g_ = nullptr;
	s_interSolverData.z_data_ = new float [64 * 100];
	s_interSolverData.z_fix_data_ = new float [64 * 100];
	s_interSolverData.batch_size_ = 64;
	s_interSolverData.d_iter_ = 25;
	s_interSolverData.main_it_ = 300;

	s_interSolverData.solver_model_file_d_.clear();
	s_interSolverData.solver_model_file_d_ = psConfigArgs->solver_d_model_;
	s_interSolverData.solver_model_file_g_.clear();
	s_interSolverData.solver_model_file_g_ = psConfigArgs->solver_g_model_;

	s_interSolverData.solver_state_file_d_.clear();
	s_interSolverData.solver_state_file_d_ = psConfigArgs->solver_d_state_;
	s_interSolverData.solver_state_file_g_.clear();
	s_interSolverData.solver_state_file_g_ = psConfigArgs->solver_g_state_;

	s_interSolverData.gpu_ones_ = nullptr;
	s_interSolverData.gpu_zeros_ = nullptr;

	generate_cuda_data(&s_interSolverData);
	//--------------------------------------------------------------------------

	current_owner = OWNER_D;

	recalculateZ(s_interSolverData.z_fix_data_ );

	pthread_barrier_init(&solvers_barrier, nullptr, 2);

	int iRC = 0;

	if ((iRC = pthread_create(&thread_d, nullptr, d_thread_fun, &s_interSolverData)) != 0)
	{
		std::cerr << "Error creating thread d " << std::endl;
		return 1;
	}

	if ((iRC = pthread_create(&thread_g, nullptr, g_thread_fun, &s_interSolverData)) != 0)
	{
		std::cerr << "Error creating thread g " << std::endl;
		return 1;
	}

	pthread_join(thread_d, nullptr);
	pthread_join(thread_g, nullptr);

	delete[] s_interSolverData.z_data_;

	CUDA_CHECK_RETURN(cudaFree(s_interSolverData.gpu_ones_));
	CUDA_CHECK_RETURN(cudaFree(s_interSolverData.gpu_zeros_));

#if 0

	caffe::Caffe::set_mode(caffe::Caffe::GPU);

	caffe::Net<float> net("./models/g.prototxt", caffe::Phase::TEST);
	net.CopyTrainedLayersFrom("./models/g.caffemodel");

//  caffe::Profiler *profiler = caffe::Profiler::Get();
//  profiler->TurnON();
//  profiler->ScopeStart("wgan");
  // random noise
  srand(time(NULL));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);
  auto input = net.blob_by_name("data");
  input->Reshape({64, 100, 1, 1});
  float *data = input->mutable_cpu_data();
  const int n = input->count();
  for (int i = 0; i < n; ++i) {
    data[i] = nd(gen);
  }
  // forward
  Timer timer;
  timer.Tic();
  net.Forward();
  timer.Toc();
  // visualization
  auto images = net.blob_by_name("gconv5");
  const int num = images->num();
  const int channels = images->channels();
  const int height = images->height();
  const int width = images->width();
  const int canvas_len = std::ceil(std::sqrt(num));
  cv::Mat canvas(canvas_len*height, canvas_len*width, CV_8UC3);
  auto clip = [](float x)->uchar {
    const int val = static_cast<int>(x*127.5 + 127.5);
    return std::max(0, std::min(255, val));
  };
  for (int i = 0; i < num; ++i) {
    const int pos_y = (i / canvas_len)*height;
    const int pos_x = (i % canvas_len)*width;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        // BGR, mxnet model saves RGB
        canvas.at<cv::Vec3b>(pos_y + y, pos_x + x)[0] = clip(images->data_at(i, 2, y, x));
        canvas.at<cv::Vec3b>(pos_y + y, pos_x + x)[1] = clip(images->data_at(i, 1, y, x));
        canvas.at<cv::Vec3b>(pos_y + y, pos_x + x)[2] = clip(images->data_at(i, 0, y, x));
      }
    }
  }
//  profiler->ScopeEnd();
//  profiler->TurnOFF();
//  profiler->DumpProfile("profile.json");
  std::cout << "generate costs " << timer.Elasped() << " ms" << std::endl;
  cv::imshow("gan-face", canvas);
  cv::waitKey();

#endif

  return 0;
}


/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}




