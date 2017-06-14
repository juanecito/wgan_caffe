/*
 * CCifar10.hpp
 *
 *  Created on: 12 jun. 2017
 *      Author: juan
 */

#ifndef SRC_CCIFAR10_HPP_
#define SRC_CCIFAR10_HPP_

#include <vector>
#include <memory>
#include <map>
#include <string>

template <typename DataType>
struct S_Cifar10_img_rgb
{
	DataType rgb_[3072];
}__attribute__((packed));

class CCifar10
{
	public:
	// /home/juan/Desktop/my_git/caffe_network/bin/cifar-10-batches-bin/
		CCifar10();
		virtual ~CCifar10();

		struct S_Cifar10_label_img
		{
			uint8_t label_;
			uint8_t red_channel_[1024];
			uint8_t green_channel_[1024];
			uint8_t blue_channel_[1024];
		}__attribute__((packed));

		void set_path(const std::string& path) {path_ = path;}

		void load_train_batch_by_index(unsigned int train_batch_index,
										struct S_Cifar10_label_img* lb_imgs,
										S_Cifar10_img_rgb<uint8_t>* img,
										uint8_t* labels);

		void load_train_batchs(void);

		void load_test_batch_by_index(unsigned int train_batch_index,
										struct S_Cifar10_label_img* lb_imgs,
										S_Cifar10_img_rgb<uint8_t>* img,
										uint8_t* labels);

		void load_test_batchs(void);

		const static std::map<uint8_t, std::string> cifar10_labels;

		uint8_t* get_train_img(unsigned int batch_index, unsigned int img_index)
		{
			struct S_Cifar10_img_rgb<uint8_t>* imgs = train_batchs_.at(batch_index).get();
			return imgs[img_index].rgb_;
		}

		uint8_t* get_test_img(unsigned int img_index)
		{
			struct S_Cifar10_img_rgb<uint8_t>* imgs = test_batchs_.at(0).get();
			return imgs[img_index].rgb_;
		}

		uint8_t* get_ori_train_img(unsigned int batch_index, unsigned int img_index)
		{
			struct S_Cifar10_label_img* imgs = ori_train_batchs_.at(batch_index).get();
			return (uint8_t*)imgs;
		}

		uint8_t* get_ori_test_img(unsigned int img_index)
		{
			struct S_Cifar10_label_img* imgs = ori_test_batchs_.at(0).get();
			return (uint8_t*)imgs;
		}

		void show_train_img(unsigned int batch_index, unsigned int img_index);
		void show_test_img(unsigned int img_index);
		void show_img(uint8_t* img, size_t img_size);

	private:

		/*!
		 *
		 */
		bool is_train_loaded_;

		bool is_test_loaded_;

		std::string path_;

		std::vector<std::shared_ptr<uint8_t> > train_labels_;
		std::vector<std::shared_ptr<uint8_t> > test_labels_;

		std::vector<std::shared_ptr<struct S_Cifar10_label_img> > ori_train_batchs_;
		std::vector<std::shared_ptr<struct S_Cifar10_label_img> > ori_test_batchs_;

		std::vector<std::shared_ptr<struct S_Cifar10_img_rgb<uint8_t> > > train_batchs_;
		std::vector<std::shared_ptr<struct S_Cifar10_img_rgb<uint8_t> > > test_batchs_;

		const static unsigned int cifar10_imgs_batch_s = 10000;
		const static unsigned int cifar10_train_batch_s = 5;
		const static unsigned int cifar10_test_batch_s = 1;

		static const std::string train_batch_pattern_name_s;
		static const std::string test_batch_pattern_name_s;

};

#endif /* SRC_CCIFAR10_HPP_ */
