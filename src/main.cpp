#include <cmath>
#include <ctime>
#include <chrono>
#include <random>
#include <memory>
#include <algorithm>
#include <caffe/caffe.hpp>
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



int main(int argc, char **argv)
{


	CCifar10 cifar10;
	cifar10.set_path("./bin/cifar-10-batches-bin");

	cifar10.load_train_batchs();
	cifar10.load_test_batchs();

	cifar10.show_img(2, 1000);

	return 0;

	caffe::Caffe::set_mode(caffe::Caffe::GPU);

	caffe::Net<float> net_g("./models/g.prototxt", caffe::Phase::TRAIN);
	caffe::Net<float> net_d("./models/d.prototxt", caffe::Phase::TRAIN);



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
