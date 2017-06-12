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

		struct S_Cifar10_img
		{
			uint8_t red_channel_[3072];
			uint8_t green_channel_[1024];
			uint8_t blue_channel_[1024];
		}__attribute__((packed));

		const static unsigned int cifar10_imgs_batch_s = 10000;
		const static unsigned int cifar10_train_batch_s = 5;
		const static unsigned int cifar10_test_batch_s = 1;

		void load_train_batch_by_index(unsigned int train_batch_index);

		void load_train_batchs(std::string pattern);

		void load_test_batch_by_index(unsigned int train_batch_index);

		void load_test_batchs(std::string pattern);

		const static std::map<uint8_t, std::string> cifar10_labels =
		{	{1, "airplane"},
			{2, "automobile"},
			{3, "bird"},
			{4, "cat"},
			{5, "deer"},
			{6, "dog"},
			{7, "frog"},
			{8, "horse"},
			{9, "ship"},
			{10, "truck"} };

		std::weak_ptr<uint8_t> get_img(unsigned int batch_index, unsigned int img_index);

	private:

		bool is_train_loaded_;

		bool is_test_loaded_;


		std::vector<uint8_t[10000]> train_labels_;
		std::vector<uint8_t[10000]> test_labels_;

		std::vector<std::shared_ptr<void>> train_batchs_;
		std::vector<std::shared_ptr<void>> test_batchs_;

};

#endif /* SRC_CCIFAR10_HPP_ */
