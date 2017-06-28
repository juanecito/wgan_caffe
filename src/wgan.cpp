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
void scale(unsigned int batch_count, unsigned int channels,
			unsigned int width, unsigned int height, float** data,
			unsigned int final_width, unsigned int final_height)
{
	unsigned int data_count = batch_count * channels * width * height;
	unsigned int new_data_count = batch_count * channels * final_width * final_height;
	float* tranf_data = new float[new_data_count];
}

////////////////////////////////////////////////////////////////////////////////
void norm(unsigned int batch_count, unsigned int channels,
			unsigned int width, unsigned int height, float** data)
{
	unsigned int data_count = batch_count * channels * width * height;

	for (unsigned int uiI = 0; uiI < data_count; uiI++)
	{
		(*data)[uiI] /= 255.0;
	}
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

int main_test(void)
{
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

#if 0


	//net.CopyTrainedLayersFrom("./models/g.caffemodel");
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

