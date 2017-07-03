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

#include <pthread.h>

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

static pthread_mutex_t solvers_mutex = PTHREAD_MUTEX_INITIALIZER;

template <typename T> void desc_network(caffe::Net<T>& net);

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
			cv::resize(img_ori, img_final, size, 2.0, 2.0, CV_INTER_LANCZOS4);//resize image
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

	caffe::Caffe::set_mode(caffe::Caffe::GPU);

	caffe::SolverParameter solver_param;
	caffe::ReadSolverParamsFromTextFileOrDie("./models/solver_d.prototxt", &solver_param);
	std::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

	caffe::MemoryDataLayer<float> *dataLayer_trainnet =
		(caffe::MemoryDataLayer<float> *) (solver->net()->layer_by_name("data").get());
	caffe::MemoryDataLayer<float> *dataLayer_testnet =
		(caffe::MemoryDataLayer<float> *) (solver->test_nets()[0]->layer_by_name("data").get());

	dataLayer_trainnet->Reset(train_imgs, train_labels, count_train);
	dataLayer_testnet->Reset(test_imgs, test_labels, count_test);

	const caffe::LayerParameter param = dataLayer_trainnet->layer_param();
	desc_network(*(solver->net().get()));
	param.PrintDebugString();

	//--------------------------------------------------------------------------
	//solver->Solve();
	solver->Step(10);
	//--------------------------------------------------------------------------

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

	caffe::Caffe::set_mode(caffe::Caffe::GPU);

	caffe::SolverParameter solver_param;
	caffe::ReadSolverParamsFromTextFileOrDie("./models/solver_g.prototxt", &solver_param);
	std::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

	auto input = solver->net()->blob_by_name("data");
	input->Reshape({64, 100, 1, 1});

	float* data = input->mutable_cpu_data();
	memcpy(data, ps_interSolverData->z_data_, 64 * 100 * sizeof(float));

	desc_network(*(solver->net().get()));

	//--------------------------------------------------------------------------
	//solver->Solve();
	solver->Step(10);
	//--------------------------------------------------------------------------

	return nullptr;
}


////////////////////////////////////////////////////////////////////////////////
int main_test(CCifar10* cifar10_data)
{
	pthread_t thread_d = 0;
	pthread_t thread_g = 0;
	struct S_InterSolverData s_interSolverData;
	s_interSolverData.cifar10_ = cifar10_data;

	srand(time(NULL));
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> nd(0, 1);
	float *z_data = new float [64 * 100];
	unsigned int n = 64 * 100 * 1 * 1;
	for (int i = 0; i < n; ++i)
	{
		z_data[i] = nd(gen);
	}

	s_interSolverData.z_data_ = z_data;

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

