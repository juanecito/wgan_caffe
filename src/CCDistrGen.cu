#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandState *state, unsigned int count)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id >= count) return;

    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}


__global__ void generate_uniform_kernel(curandState *state,
								unsigned int count,
                                float *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= count) return;
    float x;

    /* Copy state to local memory for efficiency */
    curandState localState = state[id];

    /* Generate pseudo-random uniforms */
    x = curand_uniform(&localState);

    /* Copy state back to global memory */
    state[id] = localState;

    /* Store results */
    result[id] = x;
}

__global__ void generate_normal_kernel(curandState *state,
								unsigned int count,
                                float *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= count) return;
    float x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    /* Generate pseudo-random normals */
    x = curand_normal(&localState);

    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] = x;
}


void curand_initialize(void** rand_states, unsigned int count)
{
    /* Allocate space for prng states on device */
    cudaMalloc(rand_states, count * sizeof(curandState));

	curandState *devStates = (curandState *)(*rand_states);
	
	static const int BLOCK_SIZE = 16;

    /* Setup prng states */
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE, 1);
    dim3 dimGrid((count + dimBlock.x - 1) / dimBlock.x, 1);

    setup_kernel<<<dimBlock, dimGrid>>>(devStates, count);
	
}

void curand_destroy(void* rand_states)
{
	cudaFree((curandState *)rand_states);
}


void generate_normal(void* rand_states, unsigned int count, float *result)
{
	curandState *devStates = (curandState *)rand_states;
	
	static const int BLOCK_SIZE = 16;

    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE, 1);
    dim3 dimGrid((count + dimBlock.x - 1) / dimBlock.x, 1);

    generate_normal_kernel<<<dimBlock, dimGrid>>>(devStates, count, result);
}

void generate_uniform(void* rand_states, unsigned int count, float *result)
{
	curandState *devStates = (curandState *)rand_states;
	
	static const int BLOCK_SIZE = 16;

    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE, 1);
    dim3 dimGrid((count + dimBlock.x - 1) / dimBlock.x, 1);

    generate_uniform_kernel<<<dimBlock, dimGrid>>>(devStates, count, result);
}



#if 0
int main(int argc, char *argv[])
{
    curandState *devStates;
    float *devResults, *hostResults;

    bool doubleSupported = 0;
    int device;
    struct cudaDeviceProp properties;

    /* check for double precision support */
    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaGetDeviceProperties(&properties,device));
    if ( properties.major >= 2 || (properties.major == 1 && properties.minor >= 3) ) {
        doubleSupported = 1;
    }

    unsigned int count = 6400;


    /* Allocate space for results on host */
    hostResults = (float *)calloc(count, sizeof(float));

    /* Allocate space for results on device */
    CUDA_CALL(cudaMalloc((void **)&devResults, count *
              sizeof(float)));

    /* Set results to 0 */
    CUDA_CALL(cudaMemset(devResults, 0, count *
              sizeof(float)));

    /* Allocate space for prng states on device */
    CUDA_CALL(cudaMalloc((void **)&devStates, count *
                  sizeof(curandState)));

	static const int BLOCK_SIZE = 16;

    /* Setup prng states */
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE, 1);
    dim3 dimGrid((count + dimBlock.x - 1) / dimBlock.x, 1);

    setup_kernel<<<dimBlock, dimGrid>>>(devStates, count);

    /* Set results to 0 */
    CUDA_CALL(cudaMemset(devResults, 0, count *
              sizeof(float)));

    /* Generate and use uniform pseudo-random  */
    generate_uniform_kernel<<<dimBlock, dimGrid>>>(devStates, count, devResults);

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostResults, devResults, count *
        sizeof(float), cudaMemcpyDeviceToHost));

    for (unsigned int uiI = 0; uiI < 100; ++uiI)
    {
    	std::cout << hostResults[uiI] << " " << std::endl;
    }

    /************************************************************************/
    std::cout << "-----------------------------" << std::endl;
    /* Set results to 0 */
    CUDA_CALL(cudaMemset(devResults, 0, count *
              sizeof(unsigned int)));

    /* Generate and use uniform pseudo-random  */
    generate_normal_kernel<<<dimBlock, dimGrid>>>(devStates, count, devResults);

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostResults, devResults, count *
        sizeof(float), cudaMemcpyDeviceToHost));

    for (unsigned int uiI = 0; uiI < 100; ++uiI)
    {
    	std::cout << hostResults[uiI] << " " << std::endl;
    }


    /* Cleanup */
    CUDA_CALL(cudaFree(devStates));

    CUDA_CALL(cudaFree(devResults));
    free(hostResults);
    printf("^^^^ kernel_example PASSED\n");
    return EXIT_SUCCESS;
}
#endif
