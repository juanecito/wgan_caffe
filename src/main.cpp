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

#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "CCifar10.hpp"

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

void test_1(void)
{
	//if (caffe::GPUAvailable()) {
	if (1) {
		caffe::Caffe::set_mode(caffe::Caffe::GPU);
	}
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
}

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
}

int main(int argc, char **argv)
{
	CCifar10 cifar10;
	cifar10.set_path("./bin/cifar-10-batches-bin");

	cifar10.load_train_batchs();
	cifar10.load_test_batchs();

	//--------------------------------------------------------------------------
	// Get adecuate data from cifar10
	float* train_labels = nullptr;
	cifar10.get_train_labels(0, &train_labels);

	float* test_labels = nullptr;
	cifar10.get_test_labels(&test_labels);

	float* train_imgs = nullptr;
	cifar10.get_train_batch_img(0, &train_imgs);

	float* test_imgs = nullptr;
	cifar10.get_test_batch_img(&test_imgs);
	//--------------------------------------------------------------------------
	// Test RGB image from cifar10
	//cifar10.show_train_img(3, 1500);
	//--------------------------------------------------------------------------

	caffe::Caffe::set_mode(caffe::Caffe::GPU);
	//caffe::Caffe::set_mode(caffe::Caffe::CPU);

    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie("./models/solver.prototxt", &solver_param);
    std::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

//	caffe::Net<float> net_g("./models/g.prototxt", caffe::Phase::TRAIN);
//	caffe::Net<float> net_d("./models/d.prototxt", caffe::Phase::TRAIN);

    caffe::MemoryDataLayer<float> *dataLayer_trainnet = (caffe::MemoryDataLayer<float> *) (solver->net()->layer_by_name("data").get());
    caffe::MemoryDataLayer<float> *dataLayer_testnet = (caffe::MemoryDataLayer<float> *) (solver->test_nets()[0]->layer_by_name("data").get());

//    dataLayer_trainnet->Reset(train_imgs, train_labels, CCifar10::cifar10_imgs_batch_s);
//    dataLayer_testnet->Reset(test_imgs, test_labels, CCifar10::cifar10_imgs_batch_s);

    dataLayer_trainnet->Reset(train_imgs, train_labels, 10);
    dataLayer_testnet->Reset(test_imgs, test_labels, 10);

    auto input_blob = solver->net()->blob_by_name("data");

    desc_network(*(solver->net().get()));

    return 0;

    solver->Solve();

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
