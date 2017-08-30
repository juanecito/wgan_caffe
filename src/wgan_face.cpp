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

#include "CLFWFaceDatabase.hpp"
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
													unsigned int z_data_count);

////////////////////////////////////////////////////////////////////////////////
void getZVector(const std::string& z_vector_bin_file,
								struct S_InterSolverData* ps_interSolverData);

////////////////////////////////////////////////////////////////////////////////
void show_img_CV_32FC1(unsigned int img_width, unsigned int img_height,
														const float* img_data);

////////////////////////////////////////////////////////////////////////////////
void show_grid_img_CV_32FC3(unsigned int img_width, unsigned int img_height,
							const float* img_data, unsigned int channels,
							unsigned int grid_width, unsigned int grid_height);

////////////////////////////////////////////////////////////////////////////////
void write_grid_img_CV_32FC3(const std::string& file_name,
	unsigned int img_width, unsigned int img_height, const float* img_data,
	unsigned int channels, unsigned int grid_width, unsigned int grid_height);

////////////////////////////////////////////////////////////////////////////////
unsigned int scale(unsigned int batch_count, unsigned int channels,
			unsigned int width, unsigned int height, float** data,
			unsigned int final_width, unsigned int final_height);

////////////////////////////////////////////////////////////////////////////////
unsigned int norm(unsigned int batch_count, unsigned int channels,
						unsigned int width, unsigned int height, float** data);

////////////////////////////////////////////////////////////////////////////////
unsigned int norm2(unsigned int batch_count, unsigned int channels,
						unsigned int width, unsigned int height, float** data);

////////////////////////////////////////////////////////////////////////////////
void get_data_from_faces(CLFWFaceDatabase* faces_data,
		float** train_imgs, unsigned int& count_train)
{

	auto fn_norm = [](float** data, unsigned int count) -> unsigned int {
		return norm(count, 3, 64, 64, data);
		};

	pthread_mutex_lock(&solvers_mutex);

	*train_imgs = nullptr;
	count_train = faces_data->get_imgs(train_imgs, {fn_norm});

	pthread_mutex_unlock(&solvers_mutex);
}

////////////////////////////////////////////////////////////////////////////////
static void* d_thread_fun(void* interSolverData)
{
	S_InterSolverData* ps_interSolverData = (S_InterSolverData*)interSolverData;

	float* train_imgs = nullptr;
	unsigned int count_train = 0;

	train_imgs = ps_interSolverData->train_imgs_;
	count_train = ps_interSolverData->count_train_;

	float* gpu_train_imgs = ps_interSolverData->gpu_train_imgs_;
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
			input->set_gpu_data(gpu_train_imgs + (data_index * batch_size * 3 * 64 * 64));
			input_label->set_gpu_data(ps_interSolverData->gpu_ones_);

			float errorD_real = net_d->ForwardBackward();
//			timer.tac();
//			double time1 = timer.Elasped();
//			std::cout << "Time 1: " << time1 << std::endl;
			//------------------------------------------------------------------
			// Train D with fake
//			timer.tic();
			(const_cast<std::vector<caffe::Net<float>::Callback*>&>(net_d->before_forward())).clear();

			distgen.gen_normal_dist(input_g->mutable_gpu_data());

//			timer.tac();
//			double time2 = timer.Elasped();
//			std::cout << "Time 2: " << time2 << std::endl;
//			timer.tic();

			ps_interSolverData->net_g_->Forward();
			auto blob_output_g =
					ps_interSolverData->net_g_->blob_by_name("gconv5");

			input->set_gpu_data(const_cast<float*>(blob_output_g->gpu_data()));
			input_label->set_gpu_data(ps_interSolverData->gpu_zeros_);

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

	float* train_imgs = nullptr;
	unsigned int count_train = 0;

	train_imgs = ps_interSolverData->train_imgs_;
	count_train = ps_interSolverData->count_train_;

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

		// recalculateZVector(ps_interSolverData->z_data_, batch_size, z_vector_size);
		distgen.gen_normal_dist(input_g->mutable_gpu_data());

		net_g->Forward();

		//----------------------------------------------------------------------
		// Get Fake

		cudaMemcpy(net_d_blob_data->mutable_gpu_data(),
				blob_output_g->gpu_data(),
				batch_size * 3 * 64 * 64 * sizeof(float),
				cudaMemcpyDeviceToDevice);

		input_label_d->set_gpu_data(ps_interSolverData->gpu_ones_);

		float loss_G = ps_interSolverData->net_d_->ForwardBackward();

		cudaMemcpy(blob_output_g->mutable_gpu_diff(),
				net_d_blob_data->gpu_diff(),
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

	CUDA_CHECK_RETURN(cudaMalloc((void**)&(ps_interSolverData->gpu_train_imgs_),
			ps_interSolverData->count_train_* 3 * 64 * 64 * sizeof(float)));

	CUDA_CHECK_RETURN(cudaMemcpy(ps_interSolverData->gpu_ones_, ones,
			ps_interSolverData->batch_size_ * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(ps_interSolverData->gpu_zeros_, zeros,
			ps_interSolverData->batch_size_ * sizeof(float), cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMemcpy(ps_interSolverData->gpu_train_imgs_,
			ps_interSolverData->train_imgs_,
			ps_interSolverData->count_train_* 3 * 64 * 64 * sizeof(float), cudaMemcpyHostToDevice));

	delete[] ones;
	delete[] zeros;

}

////////////////////////////////////////////////////////////////////////////////
int wgan_faces(CLFWFaceDatabase* faces_data, struct S_ConfigArgs* psConfigArgs)
{
	pthread_t thread_d = 0;
	pthread_t thread_g = 0;

	//--------------------------------------------------------------------------
	struct S_InterSolverData s_interSolverData;
	s_interSolverData.cifar10_ = nullptr;
	s_interSolverData.faces_data = faces_data;

	s_interSolverData.net_d_ = nullptr;
	s_interSolverData.net_g_ = nullptr;

	s_interSolverData.z_vector_size_ = psConfigArgs->z_vector_size_;
	s_interSolverData.batch_size_ = psConfigArgs->batch_size_;

	s_interSolverData.z_fix_data_ = new float [s_interSolverData.batch_size_ * s_interSolverData.z_vector_size_];
	cudaMallocHost((void**)&(s_interSolverData.z_data_), s_interSolverData.batch_size_ * s_interSolverData.z_vector_size_ * sizeof(float));
	s_interSolverData.gpu_z_data_ = nullptr;

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

	get_data_from_faces(s_interSolverData.faces_data,
			&(s_interSolverData.train_imgs_), s_interSolverData.count_train_);
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

	delete[] s_interSolverData.z_fix_data_;
	CUDA_CHECK_RETURN(cudaFreeHost(s_interSolverData.z_data_));

	s_interSolverData.log_file_->close();
	delete s_interSolverData.log_file_;

	CUDA_CHECK_RETURN(cudaFree(s_interSolverData.gpu_ones_));
	CUDA_CHECK_RETURN(cudaFree(s_interSolverData.gpu_zeros_));
	CUDA_CHECK_RETURN(cudaFree(s_interSolverData.gpu_train_imgs_));

	CUDA_CHECK_RETURN(cudaFree(s_interSolverData.gpu_z_fix_data_));

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

