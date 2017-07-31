/*
 * CDataSetMemMgt.hpp
 *
 *  Created on: 30 jul. 2017
 *      Author: juanitov
 */

#ifndef INCLUDE_CDATASETMEMMGT_HPP_
#define INCLUDE_CDATASETMEMMGT_HPP_

#include <caffe/blob.hpp>

template <class Dtype>
class CDataSetMemMgt
{
	public:


		CDataSetMemMgt(Dtype* train_labels, Dtype* test_labels, unsigned int label_size,
						Dtype* train_imgs, Dtype* test_imgs, unsigned int img_size,
						unsigned int m_count_train, unsigned int m_count_test,
						unsigned int block_count):
			m_train_labels(train_labels), m_test_labels(test_labels),
			m_train_imgs(train_imgs), m_test_imgs(test_imgs),
			m_count_train(m_count_train), m_count_test(m_count_test),
			m_label_size(label_size), m_img_size(img_size), m_block_count(block_count)
		{
			m_train_img_block = nullptr;
			m_test_img_block = nullptr;
			m_train_label_block = nullptr;
			m_test_label_block = nullptr;

			if (m_train_imgs != nullptr)
			{
				m_train_img_block = new caffe::Blob<Dtype>(block_count, img_size, 1, 1);
				memcpy(m_train_img_block->mutable_cpu_data(), m_train_imgs, block_count * img_size * sizeof(Dtype));
			}

			if (m_test_imgs != nullptr)
			{
				m_test_img_block = new caffe::Blob<Dtype>(block_count, img_size, 1, 1);
				memcpy(m_test_img_block->mutable_cpu_data(), m_test_imgs, block_count * img_size * sizeof(Dtype));
			}

			if (m_train_labels != nullptr)
			{
				m_train_label_block = new caffe::Blob<Dtype>(block_count, label_size, 1, 1);
				memcpy(m_train_label_block->mutable_cpu_data(), m_train_labels, block_count * label_size * sizeof(Dtype));
			}

			if (m_test_labels != nullptr)
			{
				m_test_label_block = new caffe::Blob<Dtype>(block_count, label_size, 1, 1);
				memcpy(m_test_label_block->mutable_cpu_data(), m_test_labels, block_count * label_size * sizeof(Dtype));
			}

			m_offset_start_train_block = 0;
			m_offset_end_train_block = block_count - 1;

			m_offset_start_test_block = 0;
			m_offset_end_test_block = block_count - 1;
		}

		virtual ~CDataSetMemMgt()
		{
			if (m_train_img_block != nullptr) delete m_train_img_block;
			if (m_test_img_block != nullptr) delete m_test_img_block;
			if (m_train_label_block != nullptr) delete m_train_label_block;
			if (m_test_label_block != nullptr) delete m_test_label_block;
		}

		const Dtype* get_gpu_train_img_pointer(unsigned int offset) const
		{
			const Dtype* gpu_pointer = nullptr;

			if (offset > (m_count_train * m_img_size)) return nullptr;

			if (offset >= m_offset_start_train_block && offset < m_offset_end_train_block)
			{
				gpu_pointer = m_train_img_block->gpu_data();
			}
			else
			{
				unsigned int block_index =
						((offset + m_block_count) / m_block_count) - 1;
			}

			return gpu_pointer + offset;
		}

		Dtype* get_mutable_gpu_train_img_pointer(unsigned int offset)
		{

		}

		const Dtype* get_gpu_test_img_pointer(unsigned int offset) const
		{
			return nullptr;
		}

		Dtype* get_mutable_gpu_test_img_pointer(unsigned int offset)
		{
			return nullptr;
		}

		const Dtype* get_gpu_train_label_pointer(unsigned int offset) const
		{
			return nullptr;
		}

		Dtype* get_mutable_gpu_train_label_pointer(unsigned int offset)
		{
			return nullptr;
		}

		const Dtype* get_gpu_test_label_pointer(unsigned int offset) const
		{
			return nullptr;
		}

		Dtype* get_mutable_gpu_test_label_pointer(unsigned int offset)
		{
			return nullptr;
		}


	private:

		CDataSetMemMgt() = delete;
		CDataSetMemMgt(const CDataSetMemMgt&) = delete;
		CDataSetMemMgt(CDataSetMemMgt&&) = delete;

		caffe::Blob<Dtype>* m_train_img_block;
		caffe::Blob<Dtype>* m_test_img_block;
		caffe::Blob<Dtype>* m_train_label_block;
		caffe::Blob<Dtype>* m_test_label_block;

		unsigned int m_offset_start_train_block;
		unsigned int m_offset_end_train_block;

		unsigned int m_offset_start_test_block;
		unsigned int m_offset_end_test_block;

		Dtype* m_train_labels;
		Dtype* m_test_labels;
		Dtype* m_train_imgs;
		Dtype* m_test_imgs;
		unsigned int m_count_train;
		unsigned int m_count_test;

		unsigned int m_label_size;
		unsigned int m_img_size;

		unsigned int m_block_count;

};


#endif /* INCLUDE_CDATASETMEMMGT_HPP_ */
