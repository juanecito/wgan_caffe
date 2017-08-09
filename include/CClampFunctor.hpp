/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-  */
/*
 * CClampFunctor.hpp
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

/** @file CClampFunctor.hpp
 * @author Juan Maria Gomez Lopez <juanecitorr@gmail.com>
 * @date 29 Jul 2017
 */

#ifndef INCLUDE_CCLAMPFUNCTOR_HPP_
#define INCLUDE_CCLAMPFUNCTOR_HPP_

#include "CClampFunctor.h"
#include "CTimer.hpp"

////////////////////////////////////////////////////////////////////////////////
template <typename T>
class CClampFunctor: public caffe::Net<T>::Callback
{
	public:

		CClampFunctor(caffe::Net<T>& net, T clamp_lower,T clamp_upper):
			net_(net), clamp_lower_(clamp_lower), clamp_upper_(clamp_upper),
			stream_(nullptr){}

		CClampFunctor(caffe::Net<T>& net, T clamp_lower,T clamp_upper, cudaStream_t* stream):
			net_(net), clamp_lower_(clamp_lower), clamp_upper_(clamp_upper),
			stream_(stream){}

	protected:

		virtual void run(int layer)
		{
			//CTimer timer;
			//timer.tic();
			//std::cout << "layer: " << layer << std::endl;
			std::vector<caffe::Blob<T>*>& learnable_params =
				const_cast<std::vector<caffe::Blob<T>*>& >(net_.learnable_params());
			this->clamp(learnable_params.at(layer));

			//timer.tac();
			//double time = timer.Elasped();
			//std::cout << "CClampFunctor time: " << time << " " << layer << std::endl;
		}

		virtual ~CClampFunctor(){}

		friend class caffe::Net<T>;

	private:

		void clamp(caffe::Blob<T>* blob)
		{
			bool cpu = false;

			if (cpu)
			{
				T* data = blob->mutable_cpu_data();
				unsigned int count = blob->count();

				for (unsigned int uiI = 0; uiI < count; uiI++)
				{
					if (data[uiI] < clamp_lower_){ data[uiI] = clamp_lower_; }
					else if (data[uiI] > clamp_upper_){ data[uiI] = clamp_upper_; }
				}
			}
			else
			{
				::clamp<T>(blob->count(), clamp_lower_, clamp_upper_, blob->mutable_gpu_data(), stream_);
			}
		}


		//cudaStreamSynchronize(*stream);

		caffe::Net<T>& net_;

		T clamp_lower_;
		T clamp_upper_;
		cudaStream_t* stream_;
};


#endif /* INCLUDE_CCLAMPFUNCTOR_HPP_ */
