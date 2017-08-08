/*
 * CClampFunctor.cu
 *
 *  Created on: 6 ago. 2017
 *      Author: juanitov
 */

#include "../include/CClampFunctor.h"

#include <cuda.h>
#include <cublas_v2.h>


////////////////////////////////////////////////////////////////////////////////
template<class PRECISION>
__global__ void dev_clamp(unsigned int count,
						PRECISION clamp_lower, PRECISION clamp_upper,
						PRECISION * dev_data)
{
	unsigned int uiI = blockIdx.x * blockDim.x + threadIdx.x;

	if (uiI >= count) return;

	if (dev_data[uiI] > clamp_upper)
	{
		dev_data[uiI] = clamp_upper;
	}
	else if (dev_data[uiI] < clamp_lower)
	{
		dev_data[uiI] = clamp_lower;
	}
}


////////////////////////////////////////////////////////////////////////////////
template<class PRECISION>
int clamp(unsigned int count, PRECISION clamp_lower, PRECISION clamp_upper,
			PRECISION * dev_data, cudaStream_t* stream)
{
	int BLOCK_SIZE = 16;

	dim3 dimBlockA(BLOCK_SIZE * BLOCK_SIZE, 1);
	dim3 dimGridA( (count + dimBlockA.x - 1) / (dimBlockA.x), 1);

	if (stream != nullptr)
	{
		dev_clamp<PRECISION><<<dimGridA, dimBlockA, 0, *stream>>>(
				count, clamp_lower, clamp_upper, dev_data);
	}
	else
	{
		dev_clamp<PRECISION><<<dimGridA, dimBlockA>>>(
				count, clamp_lower, clamp_upper, dev_data);
	}

	return 0;
}


template __global__ void dev_clamp<float>(unsigned int count,
						float clamp_lower, float clamp_upper,
						float * dev_data);

template __global__ void dev_clamp<double>(unsigned int count,
						double clamp_lower, double clamp_upper,
						double * dev_data);

template int clamp<float>(unsigned int count, float clamp_lower,
				float clamp_upper, float * dev_data, cudaStream_t* stream);

template int clamp<double>(unsigned int count, double clamp_lower,
		double clamp_upper, double * dev_data, cudaStream_t* stream);

