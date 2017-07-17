/*
 * CCustomLossLayerBackwardGPU.hpp
 *
 *  Created on: 16 jul. 2017
 *      Author: juanitov
 */

#ifndef INCLUDE_CCUSTOMLOSSLAYERBACKWARDGPU_HPP_
#define INCLUDE_CCUSTOMLOSSLAYERBACKWARDGPU_HPP_

#include <vector>
#include <iostream>
#include <caffe/caffe.hpp>

class CCustomLossLayerBackwardGPU
{
	public:

		CCustomLossLayerBackwardGPU(caffe::Blob<float>& extern_diff_blob):
			extern_diff_blob_(extern_diff_blob){}

		void operator()(const std::vector<caffe::Blob<float>*>& top,
			    const std::vector<bool>& propagate_down, const std::vector<caffe::Blob<float>*>& bottom)
		{
			std::cout << "CCustomLossLayerBackwardGPU " << std::endl;
			backward_gpu_g_loss(top, propagate_down, bottom);
		}


	private:

		caffe::Blob<float>& extern_diff_blob_;

		void backward_gpu_g_loss(const std::vector<caffe::Blob<float>*>& top,
				const std::vector<bool>& propagate_down, const std::vector<caffe::Blob<float>*>& bottom)
		{
			memcpy(top[0]->mutable_cpu_diff(), extern_diff_blob_.cpu_diff(), extern_diff_blob_.count() * sizeof(float));
		}

};



#endif /* INCLUDE_CCUSTOMLOSSLAYERBACKWARDGPU_HPP_ */
