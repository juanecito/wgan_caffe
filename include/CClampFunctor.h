
#ifndef INCLUDE_CCLAMPFUNCTOR_HCU_
#define INCLUDE_CCLAMPFUNCTOR_HCU_

#include <cuda.h>
#include <cublas_v2.h>

////////////////////////////////////////////////////////////////////////////////
template<class PRECISION>
int clamp(unsigned int count, PRECISION clamp_lower, PRECISION clamp_upper,
			PRECISION * dev_data, cudaStream_t* stream);

#endif // INCLUDE_CCLAMPFUNCTOR_HCU_
