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
template <typename T> void desc_network(caffe::PoolingLayer<T>& layer)
{

}

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
template <typename T>
void verify_img(caffe::Net<T>* net, CCifar10* cifar10)
{
	T* test_imgs = nullptr;
	unsigned int count_test = cifar10->get_all_test_batch_img_rgb(&test_imgs);

	T* test_labels = nullptr;
	count_test = cifar10->get_all_test_labels(&test_labels);

	S_Cifar10_img_rgb<T>* imgs_rgb_str = (S_Cifar10_img_rgb<T>*)test_imgs;

	unsigned int img_index = 3500;
	bool is_memory_data_layer = false;
	if (is_memory_data_layer)
	{
	    caffe::MemoryDataLayer<T> *dataLayer_trainnet =
	    	(caffe::MemoryDataLayer<T> *) (net->layer_by_name("cifar10").get());
		dataLayer_trainnet->Reset(imgs_rgb_str[img_index].rgb_, (T*)(test_labels + img_index), 1);
		std::cout << "label: " << test_labels[img_index] << std::endl;
	}
	else
	{
		const boost::shared_ptr<caffe::Blob<T> > sp_input = net->blob_by_name("data");

		caffe::Blob<T>* input = sp_input.get();
		input->Reshape({1, 3, 32, 32});
	//	input->Reshape({32, 32, 3, 1});
		T *data = input->mutable_cpu_data();
		const int n = input->count();

		std::cout << "n: " << n << std::endl;

		auto it = CCifar10::cifar10_labels.find(test_labels[img_index]);
		std::cout << "label: " << it->second << std::endl;

		memcpy(data, imgs_rgb_str[img_index].rgb_, 3072 * sizeof(T));
	}

	cifar10->show_test_img(img_index);

	net->ForwardBackward();

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
	return;

	//const boost::shared_ptr<caffe::Blob<T> > sh_ptr_blob = net->blob_by_name("ip1");
	const boost::shared_ptr<caffe::Blob<T> > sh_ptr_blob = net->blob_by_name("prob");
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
		std::cout << data_result[uiI] << std::endl;
	}


}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
void verify_img(caffe::Solver<T>* solver, CCifar10* cifar10)
{
	T* test_imgs = nullptr;
	unsigned int count_test = cifar10->get_all_test_batch_img_rgb(&test_imgs);

	T* test_labels = nullptr;
	count_test = cifar10->get_all_test_labels(&test_labels);

	S_Cifar10_img_rgb<T>* imgs_rgb_str = (S_Cifar10_img_rgb<T>*)test_imgs;

	unsigned int img_index = 3500;
	bool is_memory_data_layer = true;
	if (is_memory_data_layer)
	{
	    caffe::MemoryDataLayer<T> *dataLayer_trainnet =
	    	(caffe::MemoryDataLayer<T> *) (solver->net()->layer_by_name("cifar10").get());
		dataLayer_trainnet->Reset(imgs_rgb_str[img_index].rgb_, (T*)(test_labels + img_index), 1);
		std::cout << "label: " << test_labels[img_index] << std::endl;
	}
	else
	{
		const boost::shared_ptr<caffe::Blob<T> > sp_input = solver->net()->blob_by_name("data");

		caffe::Blob<T>* input = sp_input.get();
		input->Reshape({1, 3, 32, 32});
	//	input->Reshape({32, 32, 3, 1});
		T *data = input->mutable_cpu_data();
		const int n = input->count();

		std::cout << "n: " << n << std::endl;

		auto it = CCifar10::cifar10_labels.find(test_labels[img_index]);
		std::cout << "label: " << it->second << std::endl;

		memcpy(data, imgs_rgb_str[img_index].rgb_, 3072 * sizeof(T));
	}

	cifar10->show_test_img(img_index);


	solver->Step(1);

	//const boost::shared_ptr<caffe::Blob<T> > sh_ptr_blob = net->blob_by_name("ip1");
	const boost::shared_ptr<caffe::Blob<T> > sh_ptr_blob = solver->net()->blob_by_name("prob");
	caffe::Blob<T>* blob = sh_ptr_blob.get();

	T* data_result = blob->mutable_cpu_data();
	unsigned int blob_count = blob->count();
	unsigned int blob_num = blob->num();
	std::string shape_string = blob->shape_string();

	std::cout << "blob_count: " << blob_count << std::endl;
	std::cout << "blob_num: " << blob_num << std::endl;
	std::cout << "shape_string: " << blob->shape_string() << std::endl;

	for (unsigned int uiI = 0; uiI < blob_count; uiI++)
	{
		std::cout << data_result[uiI] << std::endl;
	}


}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
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

//    if (argc == 3 && strcmp(argv[1], "--test_file") == 0 )
//    {
//    	std::cout << "trained network file: " << argv[2] << std::endl;
//        caffe::Net<float> net("./models/d_1_test.prototxt", caffe::Phase::TEST);
//    	net.CopyTrainedLayersFromBinaryProto(argv[2]);
//        verify_img(&net, &cifar10);
//        return 0;
//    }
//    else if (argc == 3 && strcmp(argv[1], "--solver_file") == 0 )
//    {
//    	std::cout << "trained network file: " << argv[2] << std::endl;
//        solver->Restore(argv[2]);
////        verify_img(solver.get(), &cifar10);
////        return 0;
//    } else if (argc == 3 && strcmp(argv[1], "--train_file") == 0 )
//    {
//    	std::cout << "trained network file: " << argv[2] << std::endl;
//        solver->net()->CopyTrainedLayersFromBinaryProto(argv[2]);
//    }

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
