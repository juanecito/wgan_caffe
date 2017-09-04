/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-  */
/*
 * wgan.cpp
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

/** @file wgan.cpp
 * @author Juan Maria Gomez Lopez <juanecitorr@gmail.com>
 * @date 02 Jun 2017
 */

#include <cmath>
#include <random>
#include <memory>
#include <algorithm>
#include <iostream>
#include <functional>
#include <iomanip>
#include <vector>
#include <fstream>
#include <math.h>
#include <sys/stat.h>
#include <pthread.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

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

#include "CTimer.hpp"
#include "CCDistrGen.hpp"
#include "CClampFunctor.hpp"

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
static void takeToken(int owner)
{
	/*
	   Lock mutex and wait for signal. Note that the pthread_cond_wait
	   routine will automatically and atomically unlock mutex while it waits.
	 */
	pthread_mutex_lock(&solvers_mutex);
	while (owner != current_owner)
	{
		pthread_cond_wait(&solvers_cond, &solvers_mutex);
	}

	pthread_mutex_unlock(&solvers_mutex);

}

////////////////////////////////////////////////////////////////////////////////
static void releaseToken(int owner)
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
	}
	pthread_mutex_unlock(&solvers_mutex);
}

////////////////////////////////////////////////////////////////////////////////
void recalculateZVector(float * z_data, unsigned int batch_size,
													unsigned int z_data_count)
{
	srand(time(NULL));
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> nd(0, 1);

	unsigned int n = batch_size * z_data_count * 1 * 1;
	for (int i = 0; i < n; ++i)
	{
		z_data[i] = nd(gen);
	}
}

////////////////////////////////////////////////////////////////////////////////
bool fileExists(const std::string& file_name)
{
	struct stat st;
	if (stat(file_name.c_str(), &st) != 0)
	{
		return false;
	}
	else
	{
		if (!S_ISREG(st.st_mode))
		{
			return false;
		}
	}

	return true;
}

////////////////////////////////////////////////////////////////////////////////
bool readZVectorFromFile(const std::string& file_name,
	float * z_data, unsigned int batch_size, unsigned int z_data_count)
{
	std::fstream zVectorFile(file_name, std::ios::binary | std::ios::in);

	if (!zVectorFile.is_open()) return false;

	zVectorFile.read((char*)z_data, batch_size * z_data_count * sizeof(float));

	zVectorFile.close();

	return true;
}

////////////////////////////////////////////////////////////////////////////////
bool writeZVectorToFile(const std::string& file_name,
	const float * z_data, unsigned int batch_size, unsigned int z_data_count)
{
	std::fstream zVectorFile(file_name, std::ios::binary | std::ios::out);

	if (!zVectorFile.is_open()) return false;

	zVectorFile.write((const char*)z_data, batch_size * z_data_count * sizeof(float));

	zVectorFile.close();

	return true;
}

////////////////////////////////////////////////////////////////////////////////
void getZVector(const std::string& z_vector_bin_file,
								struct S_InterSolverData* ps_interSolverData)
{


	if (z_vector_bin_file.empty() || !fileExists(z_vector_bin_file))
	{
		recalculateZVector(ps_interSolverData->z_fix_data_,
				ps_interSolverData->batch_size_,
				ps_interSolverData->z_vector_size_);
	}

	if (!z_vector_bin_file.empty())
	{
		if (!fileExists(z_vector_bin_file))
		{
			writeZVectorToFile(z_vector_bin_file, ps_interSolverData->z_fix_data_,
								ps_interSolverData->batch_size_,
								ps_interSolverData->z_vector_size_);
		}
		else
		{
			readZVectorFromFile(z_vector_bin_file, ps_interSolverData->z_fix_data_,
								ps_interSolverData->batch_size_,
								ps_interSolverData->z_vector_size_);
		}
	}

	CUDA_CHECK_RETURN(cudaMalloc((void**)&(ps_interSolverData->gpu_z_fix_data_),
			ps_interSolverData->batch_size_ *
			ps_interSolverData->z_vector_size_ * sizeof(float)));

	CUDA_CHECK_RETURN(cudaMemcpy(ps_interSolverData->gpu_z_fix_data_,
					ps_interSolverData->z_fix_data_,
						ps_interSolverData->batch_size_ *
						ps_interSolverData->z_vector_size_ *
						sizeof(float), cudaMemcpyHostToDevice));
}

////////////////////////////////////////////////////////////////////////////////
void show_img_CV_32FC1(unsigned int img_width, unsigned int img_height,
														const float* img_data)
{
//	Test images
	const cv::Mat img(img_width, img_height, CV_32FC1, (void*)img_data);
	cv::imshow("cifar10", img);
	cv::waitKey();
}

////////////////////////////////////////////////////////////////////////////////
void show_grid_img_CV_32FC3(unsigned int img_width, unsigned int img_height,
							const float* img_data, unsigned int channels,
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

					unsigned int img_data_index =
								(y_grid * grid_width + x_grid) * img_size
													+ y_img * img_width + x_img;

					for (unsigned int c = 0; c < channels; c++)
					{
						tranf_img_data[tranf_img_data_index + c] =
							img_data[img_data_index + c * img_size_per_channel];
					}
				}
			}
		}
	}

	const cv::Mat img(img_width * grid_width, img_height * grid_height,
													CV_32FC3, tranf_img_data);
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

					unsigned int img_data_index =
									(y_grid * grid_width + x_grid) * img_size
													+ y_img * img_width + x_img;

					for (unsigned int c = 0; c < channels; c++)
					{
						tranf_img_data[tranf_img_data_index + c] =
							img_data[img_data_index + c * img_size_per_channel];
					}

				}
			}
		}
	}

	const cv::Mat img(img_width * grid_width, img_height * grid_height,
													CV_32FC3, tranf_img_data);

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
	unsigned int new_data_count =
						batch_count * channels * final_width * final_height;
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
				tranf_data + (uiI * final_size_img) +
												c * final_size_img_by_channel);
			cv::resize(img_ori, img_final, size, final_width / width,
						final_height / height, CV_INTER_LINEAR);//resize image
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
unsigned int norm2(unsigned int batch_count, unsigned int channels,
						unsigned int width, unsigned int height, float** data)
{
	for (unsigned int uiI = 0; uiI < batch_count; uiI++)
	{
		for (unsigned int uiJ = 0; uiJ < channels; uiJ++)
		{
			double mean = 0.0;
			double dev = 0.0;
			// Calculate mean by channel values
			for (unsigned int uiK = 0; uiK < (width * height); uiK++)
			{
				double X = (*data)[uiI * channels * width * height + uiJ * width * height + uiK];
				mean += X;
			}

			mean /= (double)(width * height);

			// Calculate standard_dev by channel values
			for (unsigned int uiK = 0; uiK < (width * height); uiK++)
			{
				double X = (*data)[uiI * channels * width * height + uiJ * width * height + uiK];
				dev += pow(X - mean, 2);
			}
			dev /= (double)(width * height);
			dev = sqrt(dev);

			//std::cout << "mean: " << mean << "   dev: " << dev << "     ";

			for (unsigned int uiK = 0; uiK < (width * height); uiK++)
			{
				double X = (*data)[uiI * channels * width * height + uiJ * width * height + uiK];
				double Z = (X - mean)/dev;
				double XX = Z * 0.25 + 0.5; // new mean 0.5 and new dev 0.25

				if (XX < 0.0) XX = 0.0;
				else if (XX > 1.0) XX = 1.0;

				(*data)[uiI * channels * width * height + uiJ * width * height + uiK] = XX;
			}

		}
	}

	return batch_count;
}

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

	*train_imgs = nullptr;
	count_train = cifar10->get_train_batch_img_by_label(0, train_imgs, {fn_norm, fn_scale_64});

	pthread_mutex_unlock(&solvers_mutex);
}

////////////////////////////////////////////////////////////////////////////////
void initialize_network_weights(caffe::Net<float>* net)
{
	const std::vector<std::string>& layer_names = net->layer_names();
	std::vector<caffe::Blob<float>*>& learnable_params =
		const_cast<std::vector<caffe::Blob<float>*>&>(net->learnable_params());

	srand(time(NULL));
	std::random_device rd;
	std::mt19937 gen(rd());

	std::normal_distribution<float> nd_conv(0.0, 0.001);
	std::normal_distribution<float> nd_norm(1.0, 0.02);
	std::normal_distribution<float> nd(0.0, 0.1);

	unsigned int layer_index = 0;

	for (auto it_layer_name : layer_names)
	{
		std::cout << "Layer: " << it_layer_name << std::endl;
		caffe::Blob<float>* blob = learnable_params.at(layer_index);
		const int n = blob->count();
		float* data = blob->mutable_cpu_data();

		auto layer =
			const_cast<caffe::Layer<float>* >(net->layer_by_name(it_layer_name).get());
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
		std::cout << "===========================================" << std::endl;
		std::cout << it << std::endl;
		auto blob = net->blob_by_name(it);

		unsigned int count = blob->count();
		const float* data = blob->cpu_data();
		for (unsigned int uiI = 0; uiI < count; uiI++)
		{
			if (uiI % 100 == 0) std::cout << std::endl;
			std::cout << std::setprecision(10) << data[uiI] << " ";
		}
		std::cout << std::endl << "==============================" << std::endl;
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
		std::cout << "===========================================" << std::endl;
		for (unsigned int uiI = 0; uiI < count; uiI++)
		{
			if (uiI % 100 == 0) std::cout << std::endl;
			std::cout << std::setprecision(10) << data[uiI] << " ";
		}
		std::cout << std::endl << "==============================" << std::endl;
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
		std::cout << "===========================================" << std::endl;
		for (unsigned int uiI = 0; uiI < count; uiI++)
		{
			if (uiI % 100 == 0) std::cout << std::endl;
			std::cout << std::setprecision(10) << data[uiI] << " ";
		}

	}
	std::cout << std::endl << "==================================" << std::endl;
	std::cout << std::endl << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
static void* d_thread_fun(void* interSolverData)
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

	cublasHandle_t cublasHandle = caffe::Caffe::cublas_handle();
	std::cout << "Cublas handle: " << &cublasHandle << std::endl;

	cublasStatus_t ret;
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cublasSetStream(cublasHandle, stream);

#else
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
#endif

	caffe::SolverParameter solver_param;
	boost::shared_ptr<caffe::Solver<float> > solver;

	caffe::ReadSolverParamsFromTextFileOrDie(ps_interSolverData->solver_model_file_d_, &solver_param);

	std::string snapshot_file = solver_param.snapshot_prefix();
	std::string snapshot_path =
			ps_interSolverData->output_folder_path_ + std::string("/") + snapshot_file;
	solver_param.set_snapshot_prefix(snapshot_path);

	solver.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param));

	int current_iter_d = 0;
	int max_iter_d = solver->param().max_iter();
	std::fstream& log_file = *(ps_interSolverData->log_file_);

	if (ps_interSolverData->solver_state_file_d_.size() > 0)
	{
		solver->Restore(ps_interSolverData->solver_state_file_d_.c_str());
		current_iter_d = solver->iter();

	}

	caffe::Net<float>* net_d = ps_interSolverData->net_d_ = solver->net().get();

	pthread_barrier_wait(&solvers_barrier);

	//--------------------------------------------------------------------------
	//initialize_network_weights(net_d.get());
	CClampFunctor<float>* clampFunctor = new CClampFunctor<float>(*net_d, -0.01, 0.01);

	unsigned int data_index = 0;
	unsigned int batch_size = ps_interSolverData->batch_size_;
	unsigned int z_vector_size = ps_interSolverData->z_vector_size_;

	auto input = net_d->blob_by_name("data");
	auto input_label = net_d->blob_by_name("label");
	input->Reshape({(int)batch_size, 3, 64, 64});
	input_label->Reshape({(int)batch_size, 1, 1, 1});

	auto input_g = ps_interSolverData->net_g_->blob_by_name("data");
	input_g->Reshape({(int)batch_size, (int)z_vector_size, 1, 1});

	unsigned int d_iter_by_g_real = 0;

	CTimer timer;
	CCDistrGen<float> distgen(batch_size * z_vector_size);

	for (unsigned int uiI = ps_interSolverData->current_iter_;
			uiI < ps_interSolverData->max_iter_; uiI++)
	{
		// Discriminator and generator threads synchronization
		takeToken(OWNER_D);

		if (uiI < 25 || uiI % 500 == 0){ d_iter_by_g_real = 100; }
		else { d_iter_by_g_real = ps_interSolverData->d_iters_by_g_iter_; }

		for (unsigned int uiJ = 0; uiJ < d_iter_by_g_real; uiJ++)
		{
//			timer.tic();
			if ((data_index * batch_size) > (count_train - batch_size) ) data_index = 0;

			//------------------------------------------------------------------
			// Train D with real
			net_d->add_before_forward(clampFunctor);

			float* data_d = input->mutable_cpu_data();
			memcpy(data_d, train_imgs + (data_index * batch_size * 3 * 64 * 64),
					batch_size * 3 * 64 * 64 * sizeof(float));

			cudaMemcpy(input_label->mutable_gpu_data(),
						ps_interSolverData->gpu_ones_,
						ps_interSolverData->batch_size_ * sizeof(float),
						cudaMemcpyDeviceToDevice);

			float errorD_real = net_d->ForwardBackward();

			//------------------------------------------------------------------
			// Train D with fake
			(const_cast<std::vector<caffe::Net<float>::Callback*>&>(net_d->before_forward())).clear();

			recalculateZVector(ps_interSolverData->z_data_,
								batch_size, z_vector_size);

			float* data_g = input_g->mutable_cpu_data();
			memcpy(data_g, ps_interSolverData->z_data_,
					batch_size * z_vector_size * sizeof(float));
			ps_interSolverData->net_g_->Forward();
			auto blob_output_g =
					ps_interSolverData->net_g_->blob_by_name("gconv5");

			cudaMemcpy(input->mutable_gpu_data(),
				blob_output_g->gpu_data(),
				batch_size * 3 * 64 * 64 * sizeof(float),
				cudaMemcpyDeviceToDevice);

			cudaMemcpy(input_label->mutable_gpu_data(),
					ps_interSolverData->gpu_zeros_, batch_size * sizeof(float),
					cudaMemcpyDeviceToDevice);
					
//			timer.tac();
//			double time3 = timer.Elasped();
//			std::cout << "Time 3: " << time3 << std::endl;
//			timer.tic();

			solver->StepOne_ForBackAndUpdate();

//			timer.tac();
//			double time4 = timer.Elasped();
//			std::cout << "Time 4: " << time4 << std::endl;

			if (uiJ == (d_iter_by_g_real - 1))
			{
				float errorD_fake = 0.0;
				errorD_fake = net_d->blob_by_name("loss")->cpu_data()[0];


				std::cout << "iteration:" << uiI << ";";
				std::cout << "errorD_real:" << errorD_real << ";";
				std::cout << "errorD_fake:" << errorD_fake << ";";
				float errorD = errorD_real + errorD_fake;
				std::cout << "errorD:" << errorD << ";";

				log_file << "iteration:" << uiI << ";";
				log_file << "errorD_real:" << errorD_real << ";";
				log_file << "errorD_fake:" << errorD_fake << ";";
				log_file << "errorD:" << errorD << ";";
				log_file.flush();
			}
			net_d->ClearParamDiffs();
			data_index++;
		}

		if (uiI > ps_interSolverData->current_iter_ && uiI == (ps_interSolverData->main_iters_ - 1))
		{
			solver->Snapshot();
		}

		releaseToken(OWNER_D);
	}

	//--------------------------------------------------------------------------
	pthread_barrier_wait(&solvers_barrier);

	return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
static void* g_thread_fun(void* interSolverData)
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

	cublasHandle_t cublasHandle = caffe::Caffe::cublas_handle();
	std::cout << "Cublas handle: " << &cublasHandle << std::endl;

	cublasStatus_t ret;
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cublasSetStream(cublasHandle, stream);

#else
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
#endif

	std::fstream& log_file = *(ps_interSolverData->log_file_);

	caffe::SolverParameter solver_param;
	caffe::ReadSolverParamsFromTextFileOrDie(ps_interSolverData->solver_model_file_g_, &solver_param);

	std::string snapshot_file = solver_param.snapshot_prefix();
	std::string snapshot_path =
			ps_interSolverData->output_folder_path_ + std::string("/") + snapshot_file;
	solver_param.set_snapshot_prefix(snapshot_path);

	std::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

	int current_iter_g = 0;
	int max_iter_g = solver->param().max_iter();
	if (ps_interSolverData->solver_state_file_g_.size() > 0)
	{
		solver->Restore(ps_interSolverData->solver_state_file_g_.c_str());
		current_iter_g = solver->iter();
	}

	ps_interSolverData->current_iter_ = current_iter_g;
	ps_interSolverData->max_iter_ = max_iter_g;

	caffe::Net<float>* net_g = ps_interSolverData->net_g_ = solver->net().get();
	pthread_barrier_wait(&solvers_barrier);
	//--------------------------------------------------------------------------
	unsigned int batch_size = ps_interSolverData->batch_size_;
	unsigned int z_vector_size = ps_interSolverData->z_vector_size_;

	CCDistrGen<float> distgen(batch_size * z_vector_size);

	auto input_g = ps_interSolverData->net_g_->blob_by_name("data");
	input_g->Reshape({(int)batch_size, (int)z_vector_size, 1, 1});

	auto blob_output_g = net_g->blob_by_name("gconv5");
	auto net_d_blob_data = ps_interSolverData->net_d_->blob_by_name("data");
	auto input_label_d = ps_interSolverData->net_d_->blob_by_name("label");

	for (unsigned int uiI = current_iter_g; uiI < max_iter_g; uiI++)
	{
		takeToken(OWNER_G);

		recalculateZVector(ps_interSolverData->z_data_, batch_size, z_vector_size);

		memcpy(input_g->mutable_cpu_data(), ps_interSolverData->z_data_,
				batch_size * z_vector_size * sizeof(float));
		net_g->Forward();

		//----------------------------------------------------------------------
		// Get Fake

		cudaMemcpy(net_d_blob_data->mutable_gpu_data(),
				blob_output_g->gpu_data(),
				batch_size * 3 * 64 * 64 * sizeof(float),
				cudaMemcpyDeviceToDevice);

		cudaMemcpy(input_label_d->mutable_gpu_data(),
					ps_interSolverData->gpu_ones_,
					batch_size * sizeof(float),
					cudaMemcpyDeviceToDevice);

		float loss_G = ps_interSolverData->net_d_->ForwardBackward();

		cudaMemcpy(blob_output_g->mutable_gpu_diff(),
				net_d_blob_data->gpu_diff(),
				batch_size * 3 * 64 * 64 * sizeof(float),
				cudaMemcpyDeviceToDevice);

		cudaMemcpy(blob_output_g->mutable_gpu_data(),
				net_d_blob_data->gpu_data(),
				batch_size * 3 * 64 * 64 * sizeof(float),
				cudaMemcpyDeviceToDevice);

		solver->StepOne_BackAndUpdate();

		std::cout << "loss_G:" << loss_G << std::endl;
		log_file << "loss_G:" << loss_G << std::endl;

		if (uiI > 0 && uiI % 100 == 0)
		{
			cudaMemcpy(input_g->mutable_gpu_data(),
								ps_interSolverData->gpu_z_fix_data_,
								batch_size * z_vector_size * sizeof(float),
								cudaMemcpyDeviceToDevice);

			net_g->Forward();
			const float* img_g_data = net_g->blob_by_name("gconv5")->cpu_data();
			std::string file_name = ps_interSolverData->output_folder_path_ + std::string("/") +
				std::string("wgan_grid") + std::to_string(uiI) + std::string(".yml");
			write_grid_img_CV_32FC3(file_name, 64, 64, img_g_data, 3, 8, 8);
		}

		net_g->ClearParamDiffs();
		ps_interSolverData->net_d_->ClearParamDiffs();
		releaseToken(OWNER_G);
	}

	pthread_barrier_wait(&solvers_barrier);
	return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
static void generate_cuda_data(S_InterSolverData* ps_interSolverData)
{
	float* ones = new float [ps_interSolverData->batch_size_];
	float* zeros = new float [ps_interSolverData->batch_size_];

	for (unsigned int uiI = 0; uiI < ps_interSolverData->batch_size_; uiI++)
	{
		ones[uiI] = 1.0;
		zeros[uiI] = 0.0;
	}

	CUDA_CHECK_RETURN(cudaMalloc((void**)&(ps_interSolverData->gpu_ones_),
			ps_interSolverData->batch_size_ * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&(ps_interSolverData->gpu_zeros_),
			ps_interSolverData->batch_size_ * sizeof(float)));

//	CUDA_CHECK_RETURN(cudaMalloc((void**)&(ps_interSolverData->gpu_train_imgs_),
//			ps_interSolverData->count_train_* 3 * 64 * 64 * sizeof(float)));

	CUDA_CHECK_RETURN(cudaMemcpy(ps_interSolverData->gpu_ones_, ones,
			ps_interSolverData->batch_size_ * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(ps_interSolverData->gpu_zeros_, zeros,
			ps_interSolverData->batch_size_ * sizeof(float), cudaMemcpyHostToDevice));

//	CUDA_CHECK_RETURN(cudaMemcpy(ps_interSolverData->gpu_train_imgs_,
//			ps_interSolverData->train_imgs_,
//			ps_interSolverData->count_train_* 3 * 64 * 64 * sizeof(float), cudaMemcpyHostToDevice));

	delete[] ones;
	delete[] zeros;

}

////////////////////////////////////////////////////////////////////////////////
int wgan(CCifar10* cifar10_data, struct S_ConfigArgs* psConfigArgs)
{
	pthread_t thread_d = 0;
	pthread_t thread_g = 0;

	//--------------------------------------------------------------------------
	struct S_InterSolverData s_interSolverData;
	s_interSolverData.cifar10_ = cifar10_data;
	s_interSolverData.faces_data = nullptr;

	s_interSolverData.net_d_ = nullptr;
	s_interSolverData.net_g_ = nullptr;

	s_interSolverData.z_vector_size_ = psConfigArgs->z_vector_size_;
	s_interSolverData.batch_size_ = psConfigArgs->batch_size_;

	s_interSolverData.z_data_ = new float [s_interSolverData.batch_size_ * s_interSolverData.z_vector_size_];
	s_interSolverData.z_fix_data_ = new float [s_interSolverData.batch_size_ * s_interSolverData.z_vector_size_];
	s_interSolverData.d_iters_by_g_iter_ = psConfigArgs->d_iters_by_g_iter_;
	s_interSolverData.main_iters_ = psConfigArgs->main_iters_;

	s_interSolverData.current_iter_ = 0;
	s_interSolverData.max_iter_ = 0;

	s_interSolverData.solver_model_file_d_.clear();
	s_interSolverData.solver_model_file_d_ = psConfigArgs->solver_d_model_;
	s_interSolverData.solver_model_file_g_.clear();
	s_interSolverData.solver_model_file_g_ = psConfigArgs->solver_g_model_;

	s_interSolverData.solver_state_file_d_.clear();
	s_interSolverData.solver_state_file_d_ = psConfigArgs->solver_d_state_;
	s_interSolverData.solver_state_file_g_.clear();
	s_interSolverData.solver_state_file_g_ = psConfigArgs->solver_g_state_;

	s_interSolverData.log_file_ = new std::fstream(psConfigArgs->logarg_, std::ios_base::app);
	s_interSolverData.output_folder_path_ = psConfigArgs->output_folder_path_;
	s_interSolverData.gpu_ones_ = nullptr;
	s_interSolverData.gpu_zeros_ = nullptr;

	generate_cuda_data(&s_interSolverData);
	//--------------------------------------------------------------------------

	getZVector(psConfigArgs->z_vector_bin_file_, &s_interSolverData);

	current_owner = OWNER_D;

	pthread_barrier_init(&solvers_barrier, nullptr, 2);

	int iRC = 0;

	if ((iRC = pthread_create(&thread_d, nullptr,
									d_thread_fun, &s_interSolverData)) != 0)
	{
		std::cerr << "Error creating thread d " << std::endl;
		return 1;
	}

	if ((iRC = pthread_create(&thread_g, nullptr,
									g_thread_fun, &s_interSolverData)) != 0)
	{
		std::cerr << "Error creating thread g " << std::endl;
		return 1;
	}

	pthread_join(thread_d, nullptr);
	pthread_join(thread_g, nullptr);

	delete[] s_interSolverData.z_data_;
	delete[] s_interSolverData.z_fix_data_;

	s_interSolverData.log_file_->close();
	delete s_interSolverData.log_file_;

	CUDA_CHECK_RETURN(cudaFree(s_interSolverData.gpu_ones_));
	CUDA_CHECK_RETURN(cudaFree(s_interSolverData.gpu_zeros_));

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line,
									const char *statement, cudaError_t err)
{
	if (err == cudaSuccess) return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err)
			<< "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

