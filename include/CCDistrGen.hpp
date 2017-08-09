/*
 * CCDistrGen.hpp
 *
 *  Created on: 9 ago. 2017
 *      Author: juan
 */

#ifndef INCLUDE_CCDISTRGEN_HPP_
#define INCLUDE_CCDISTRGEN_HPP_

#include "CCDistrGen.h"

template <class T>
class CCDistrGen
{
	public:

		CCDistrGen(unsigned int count = 64)
		{
			rand_states_ = nullptr;
			count_ = count;
			curand_initialize(&rand_states_, count_);
		}

		~CCDistrGen()
		{
			curand_destroy(rand_states_);
		}

		void gen_normal_dist(float *result)
		{
			generate_normal(rand_states_, count_, result);
		}

		void gen_uniform_dist(float *result)
		{
			generate_uniform(rand_states_, count_, result);
		}

	private:

		void* rand_states_;
		unsigned int count_;
};



#endif /* INCLUDE_CCDISTRGEN_HPP_ */
