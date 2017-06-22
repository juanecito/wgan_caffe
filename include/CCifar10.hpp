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

#include <cuda.h>

/**
 *
 */
template <typename DataType>
struct S_Cifar10_img_rgb
{
	DataType rgb_[3072];
}__attribute__((packed));

/**
 *
 */
struct S_Cifar10_label_img
{
	uint8_t label_;
	uint8_t red_channel_[1024];
	uint8_t green_channel_[1024];
	uint8_t blue_channel_[1024];
}__attribute__((packed));

template <typename DataType>
struct S_Cifar10_img
{
	DataType red_channel_[1024];
	DataType green_channel_[1024];
	DataType blue_channel_[1024];
}__attribute__((packed));

////////////////////////////////////////////////////////////////////////////////
/**
 *
 */
class CCifar10
{
	public:

		CCifar10();
		virtual ~CCifar10();


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

		//----------------------------------------------------------------------
		template <typename T>
		size_t get_batch_imgs_size(void) const
		{
			return sizeof(struct S_Cifar10_img<T>) * cifar10_imgs_batch_s;
		}

		template <typename T>
		size_t get_batch_labels_size(void) const
		{
			return sizeof(T) * cifar10_imgs_batch_s;
		}
		//----------------------------------------------------------------------
		// Get RGB images
		/**
		 *
		 * @param batch_index
		 * @param img_index
		 * @return
		 */
		uint8_t* get_train_img_rgb(unsigned int batch_index, unsigned int img_index)
		{
			struct S_Cifar10_img_rgb<uint8_t>* imgs = train_batchs_.at(batch_index).get();
			return imgs[img_index].rgb_;
		}

		/**
		 *
		 * @param img_index
		 * @return
		 */
		uint8_t* get_test_img_rgb(unsigned int img_index)
		{
			struct S_Cifar10_img_rgb<uint8_t>* imgs = test_batchs_.at(0).get();
			return imgs[img_index].rgb_;
		}
		//----------------------------------------------------------------------
		// Get images without labels
		/**
		 *
		 * @param batch_index
		 * @return
		 */
		template <typename T>
		void get_train_batch_img(unsigned int batch_index, T** imgs);

		template <typename T>
		unsigned int get_all_train_batch_img(T** imgs);

		template <typename T>
		unsigned int get_all_train_batch_img_rgb(T** imgs);

		/**
		 *
		 * @return
		 */
		template <typename T>
		void get_test_batch_img(T** imgs);

		template <typename T>
		unsigned int get_all_test_batch_img(T** imgs);

		template <typename T>
		unsigned int get_all_test_batch_img_rgb(T** imgs);
		//----------------------------------------------------------------------
		// Get images with label, as we can find in cifar10 files
		/**
		 *
		 * @param batch_index
		 * @param img_index
		 * @return
		 */
		uint8_t* get_ori_train_img(unsigned int batch_index, unsigned int img_index)
		{
			struct S_Cifar10_label_img* imgs = ori_train_batchs_.at(batch_index).get();
			return (uint8_t*)imgs;
		}

		//----------------------------------------------------------------------
		// Get labels
		void get_train_labels(unsigned int batch_index, uint8_t** t_labels)
		{
			*t_labels = new uint8_t [cifar10_imgs_batch_s];
			memcpy(*t_labels, this->train_labels_.at(batch_index).get(),
							sizeof(uint8_t) * cifar10_imgs_batch_s);
		}

		void get_test_labels(uint8_t** t_labels)
		{
			*t_labels = new uint8_t [cifar10_imgs_batch_s];
			memcpy(*t_labels, this->test_labels_.at(0).get(),
							sizeof(uint8_t) * cifar10_imgs_batch_s);
		}

		template <typename T>
		void get_train_labels(unsigned int batch_index, T** t_labels);

		template <typename T>
		unsigned int get_all_train_labels(T** t_labels);

		template <typename T>
		void get_test_labels(T** t_labels);
		template <typename T>
		unsigned int get_all_test_labels(T** t_labels);
		//----------------------------------------------------------------------
		/**
		 *
		 * @param img_index
		 * @return
		 */
		uint8_t* get_ori_test_img(unsigned int img_index)
		{
			struct S_Cifar10_label_img* imgs = ori_test_batchs_.at(0).get();
			return (uint8_t*)imgs;
		}

		void show_train_img(unsigned int batch_index, unsigned int img_index);
		void show_test_img(unsigned int img_index);
		void show_img(uint8_t* img, size_t img_size);
		static void print_cifar10_labels(void)
		{
			for (auto it : CCifar10::cifar10_labels)
			{
				std::cout << "(" << (int)(it.first) << ":" << it.second << ") ";
			}
			std::cout << std::endl;
		}

		const static unsigned int cifar10_imgs_batch_s = 10000;
		const static unsigned int cifar10_train_batch_s = 5;
		const static unsigned int cifar10_test_batch_s = 1;

		static const std::string train_batch_pattern_name_s;
		static const std::string test_batch_pattern_name_s;
		const static std::map<uint8_t, std::string> cifar10_labels;

	private:

		void calculate_means(void);

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

		struct S_Cifar10_img<double> mean_values_;
};



////////////////////////////////////////////////////////////////////////////////
template <typename T>
void CCifar10::get_train_batch_img(unsigned int batch_index, T** imgs)
{
	struct S_Cifar10_img<T>* tmp_img = new struct S_Cifar10_img<T> [cifar10_imgs_batch_s];
	struct S_Cifar10_label_img* ori_imgs = ori_train_batchs_.at(batch_index).get();

	for (unsigned int uiI = 0; uiI < cifar10_imgs_batch_s; uiI++)
	{
		for (unsigned int uiJ = 0; uiJ < 1024; uiJ++)
		{
			tmp_img[uiI].red_channel_[uiJ] = ori_imgs[uiJ].red_channel_[uiJ];
			tmp_img[uiI].green_channel_[uiJ] = ori_imgs[uiJ].green_channel_[uiJ];
			tmp_img[uiI].blue_channel_[uiJ] = ori_imgs[uiJ].blue_channel_[uiJ];
		}
	}
	*imgs = (T*)(ori_imgs);
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
unsigned int CCifar10::get_all_train_batch_img(T** imgs)
{
	unsigned int count = cifar10_imgs_batch_s * ori_train_batchs_.size();

	struct S_Cifar10_img<T>* tmp_img = new struct S_Cifar10_img<T> [count];

	unsigned int batch_index = 0;
	for (auto it : ori_train_batchs_)
	{
		struct S_Cifar10_label_img* ori_imgs = it.get();

		for (unsigned int uiI = 0; uiI < cifar10_imgs_batch_s; uiI++)
		{
			for (unsigned int uiJ = 0; uiJ < 1024; uiJ++)
			{
				tmp_img[(batch_index * cifar10_imgs_batch_s) + uiI].red_channel_[uiJ] = ori_imgs[uiI].red_channel_[uiJ];
				tmp_img[(batch_index * cifar10_imgs_batch_s) + uiI].green_channel_[uiJ] = ori_imgs[uiI].green_channel_[uiJ];
				tmp_img[(batch_index * cifar10_imgs_batch_s) + uiI].blue_channel_[uiJ] = ori_imgs[uiI].blue_channel_[uiJ];
			}
		}
		batch_index++;
	}

	*imgs = (T*)(tmp_img);

	return count;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
unsigned int CCifar10::get_all_train_batch_img_rgb(T** imgs)
{
	unsigned int count = cifar10_imgs_batch_s * ori_train_batchs_.size();
	struct S_Cifar10_img_rgb<T>* tmp_img = new struct S_Cifar10_img_rgb<T> [count];

	unsigned int batch_index = 0;
	for (auto it : ori_train_batchs_)
	{
		struct S_Cifar10_label_img* ori_imgs = it.get();

		for (unsigned int uiI = 0; uiI < cifar10_imgs_batch_s; uiI++)
		{
			for (unsigned int uiJ = 0; uiJ < 1024; uiJ++)
			{
				tmp_img[(batch_index * cifar10_imgs_batch_s) + uiI].rgb_[uiJ * 3] = ori_imgs[uiI].red_channel_[uiJ];
				tmp_img[(batch_index * cifar10_imgs_batch_s) + uiI].rgb_[uiJ * 3 + 1] = ori_imgs[uiI].green_channel_[uiJ];
				tmp_img[(batch_index * cifar10_imgs_batch_s) + uiI].rgb_[uiJ * 3 + 2] = ori_imgs[uiI].blue_channel_[uiJ];
			}
		}
		batch_index++;
	}

	*imgs = (T*)(tmp_img);

	return count;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
void CCifar10::get_test_batch_img(T** imgs)
{
	struct S_Cifar10_img<T>* tmp_img = new struct S_Cifar10_img<T> [cifar10_imgs_batch_s];
	struct S_Cifar10_label_img* ori_imgs = ori_test_batchs_.at(0).get();

	for (unsigned int uiI = 0; uiI < cifar10_imgs_batch_s; uiI++)
	{
		for (unsigned int uiJ = 0; uiJ < 1024; uiJ++)
		{
			tmp_img[uiI].red_channel_[uiJ] = ori_imgs[uiI].red_channel_[uiJ];
			tmp_img[uiI].green_channel_[uiJ] = ori_imgs[uiI].green_channel_[uiJ];
			tmp_img[uiI].blue_channel_[uiJ] = ori_imgs[uiI].blue_channel_[uiJ];
		}
	}
	*imgs = (T*)(ori_imgs);
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
unsigned int CCifar10::get_all_test_batch_img(T** imgs)
{
	unsigned int count = cifar10_imgs_batch_s * ori_test_batchs_.size();

	struct S_Cifar10_img<T>* tmp_img = new struct S_Cifar10_img<T> [count];

	unsigned int batch_index = 0;
	for (auto it : ori_test_batchs_)
	{
		struct S_Cifar10_label_img* ori_imgs = it.get();

		for (unsigned int uiI = 0; uiI < cifar10_imgs_batch_s; uiI++)
		{
			for (unsigned int uiJ = 0; uiJ < 1024; uiJ++)
			{
				tmp_img[(batch_index * cifar10_imgs_batch_s) + uiI].red_channel_[uiJ] = ori_imgs[uiI].red_channel_[uiJ];
				tmp_img[(batch_index * cifar10_imgs_batch_s) + uiI].green_channel_[uiJ] = ori_imgs[uiI].green_channel_[uiJ];
				tmp_img[(batch_index * cifar10_imgs_batch_s) + uiI].blue_channel_[uiJ] = ori_imgs[uiI].blue_channel_[uiJ];
			}
		}
		batch_index++;
	}

	*imgs = (T*)(tmp_img);

	return count;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
unsigned int CCifar10::get_all_test_batch_img_rgb(T** imgs)
{
	unsigned int count = cifar10_imgs_batch_s * ori_test_batchs_.size();
	struct S_Cifar10_img_rgb<T>* tmp_img = new struct S_Cifar10_img_rgb<T> [count];

	unsigned int batch_index = 0;
	for (auto it : ori_test_batchs_)
	{
		struct S_Cifar10_label_img* ori_imgs = it.get();

		for (unsigned int uiI = 0; uiI < cifar10_imgs_batch_s; uiI++)
		{
			for (unsigned int uiJ = 0; uiJ < 1024; uiJ++)
			{
				tmp_img[(batch_index * cifar10_imgs_batch_s) + uiI].rgb_[uiJ * 3] = ori_imgs[uiI].red_channel_[uiJ];
				tmp_img[(batch_index * cifar10_imgs_batch_s) + uiI].rgb_[uiJ * 3 + 1] = ori_imgs[uiI].green_channel_[uiJ];
				tmp_img[(batch_index * cifar10_imgs_batch_s) + uiI].rgb_[uiJ * 3 + 2] = ori_imgs[uiI].blue_channel_[uiJ];
			}
		}
		batch_index++;
	}

	*imgs = (T*)(tmp_img);

	return count;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
void CCifar10::get_train_labels(unsigned int batch_index, T** t_labels)
{
	*t_labels = new T [cifar10_imgs_batch_s];
	uint8_t* train_label = this->train_labels_.at(batch_index).get();
	for (unsigned int uiI = 0; uiI < cifar10_imgs_batch_s; uiI++)
	{
		(*t_labels)[uiI] = train_label[uiI];
	}
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
void CCifar10::get_test_labels(T** t_labels)
{
	*t_labels = new T [cifar10_imgs_batch_s];
	uint8_t* test_label = this->train_labels_.at(0).get();
	for (unsigned int uiI = 0; uiI < cifar10_imgs_batch_s; uiI++)
	{
		(*t_labels)[uiI] = test_label[uiI];
	}
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
unsigned int CCifar10::get_all_train_labels(T** t_labels)
{
	unsigned int count = cifar10_imgs_batch_s * train_labels_.size();
	*t_labels = new T [count];

	unsigned int batch_index = 0;
	for (auto it : train_labels_)
	{
		for (unsigned int uiI = 0; uiI < cifar10_imgs_batch_s; uiI++)
		{
			(*t_labels)[batch_index * cifar10_imgs_batch_s + uiI] = it.get()[uiI];
		}
		batch_index++;
	}

	return count;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
unsigned int CCifar10::get_all_test_labels(T** t_labels)
{
	unsigned int count = cifar10_imgs_batch_s * test_labels_.size();
	*t_labels = new T [count];

	unsigned int batch_index = 0;
	for (auto it : test_labels_)
	{
		for (unsigned int uiI = 0; uiI < cifar10_imgs_batch_s; uiI++)
		{
			(*t_labels)[batch_index * cifar10_imgs_batch_s + uiI] = it.get()[uiI];
		}
		batch_index++;
	}

	return count;
}


#endif /* SRC_CCIFAR10_HPP_ */
