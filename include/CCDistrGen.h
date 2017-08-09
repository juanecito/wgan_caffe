/*
 * CCDistrGen.h
 *
 *  Created on: 9 ago. 2017
 *      Author: juan
 */

#ifndef INCLUDE_CCDISTRGEN_H_
#define INCLUDE_CCDISTRGEN_H_

void curand_initialize(void** rand_states, unsigned int count);
void curand_destroy(void* rand_states);


void generate_normal(void* rand_states, unsigned int count, float *result);

void generate_uniform(void* rand_states, unsigned int count, float *result);


#endif /* INCLUDE_CCDISTRGEN_H_ */
