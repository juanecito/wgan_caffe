/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-  */
/*
 * CCustomLossLayerBackwardGPU.hpp
 * Copyright (C) 2017 Juan Maria Gomez Lopez <juanecitorr@gmail.com>
 *
 * caffe_network is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * caffe_wgan is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/** @file CCustomLossLayerBackwardGPU.hpp
 * @author Juan Maria Gomez Lopez <juanecitorr@gmail.com>
 * @date 16 Jul 2017
 */

#ifndef INCLUDE_CCUSTOMLOSSLAYERBACKWARDGPU_HPP_
#define INCLUDE_CCUSTOMLOSSLAYERBACKWARDGPU_HPP_

#include <vector>
#include <iostream>
#include <caffe/caffe.hpp>

template <class Dtype>
class CCustomLossLayerBackwardGPU
{
	public:

		CCustomLossLayerBackwardGPU(caffe::Blob<Dtype>& extern_diff_blob):
			extern_diff_blob_(extern_diff_blob){}

		void operator()(const std::vector<caffe::Blob<Dtype>*>& top,
			    const std::vector<bool>& propagate_down, const std::vector<caffe::Blob<Dtype>*>& bottom)
		{
			std::cout << "CCustomLossLayerBackwardGPU " << std::endl;
			backward_gpu_g_loss(top, propagate_down, bottom);
		}


	private:

		caffe::Blob<Dtype>& extern_diff_blob_;

		void backward_gpu_g_loss(const std::vector<caffe::Blob<Dtype>*>& top,
				const std::vector<bool>& propagate_down, const std::vector<caffe::Blob<Dtype>*>& bottom)
		{
			memcpy(top[0]->mutable_cpu_diff(), extern_diff_blob_.cpu_diff(), extern_diff_blob_.count() * sizeof(Dtype));
		}

};

////////////////////////////////////////////////////////////////////////////////
template <class Dtype>
void forward_cpu_g_loss(const std::vector<caffe::Blob<Dtype>*>&,
  	  	  const std::vector<caffe::Blob<Dtype>*>&)
{
	// TODO:
	std::cout << "forward_cpu_g_loss " << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
template <class Dtype>
void forward_gpu_g_loss(const std::vector<caffe::Blob<Dtype>*>&,
  	  	  const std::vector<caffe::Blob<Dtype>*>&)
{
	// TODO:
	std::cout << "forward_gpu_g_loss " << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
template <class Dtype>
void backward_cpu_g_loss(const std::vector<caffe::Blob<Dtype>*>&,
						const std::vector<bool>&,
						const std::vector<caffe::Blob<Dtype>*>&)
{
	// TODO:
	std::cout << "backward_cpu_g_loss " << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
template <class Dtype>
void backward_gpu_g_loss(const std::vector<caffe::Blob<Dtype>*>&,
						const std::vector<bool>&,
						const std::vector<caffe::Blob<Dtype>*>&)
{
	// TODO:
	std::cout << "backward_gpu_g_loss " << std::endl;
}


#endif /* INCLUDE_CCUSTOMLOSSLAYERBACKWARDGPU_HPP_ */
