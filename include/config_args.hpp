/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-  */
/*
 * config_args.hpp
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

/** @file config_args.hpp
 * @author Juan Maria Gomez Lopez <juanecitorr@gmail.com>
 * @date 27 Jul 2017
 */

#ifndef INCLUDE_CONFIG_ARGS_HPP_
#define INCLUDE_CONFIG_ARGS_HPP_

#include <string>
#include <fstream>

#include <caffe/caffe.hpp>

#include "CCifar10.hpp"
#include "CLFWFaceDatabase.hpp"


////////////////////////////////////////////////////////////////////////////////
enum E_CONFIG_ARGS_OPTS
{
	OPT_LOG,
	OPT_DATASET,
	OPT_PRE_TRAIN_FILE,
	OPT_WGAN_D_SOLVER_MODEL_FILE,
	OPT_WGAN_G_SOLVER_MODEL_FILE,
	OPT_WGAN_D_SOLVER_STATE_FILE,
	OPT_WGAN_G_SOLVER_STATE_FILE,
	OPT_DATA_SOURCE_FOLDER_PATH,
	OPT_OUTPUT_FOLDER_PATH,
	OPT_D_ITERS_BY_G_ITER,
	OPT_MAIN_ITERS,
	OPT_BATCH_SIZE,
	OPT_Z_VECTOR_SIZE,
	OPT_Z_VECTOR_BIN_FILE
};

////////////////////////////////////////////////////////////////////////////////
struct S_ConfigArgs
{
	char* logarg_;
	char* pretrain_file_name_;

	int run_wgan_;
	int run_cifar10_training_;
	int test_cifar10_;

	std::string solver_d_model_;
	std::string solver_g_model_;
	std::string solver_d_state_;
	std::string solver_g_state_;

	std::string dataset_;
	std::string data_source_folder_path_;
	std::string output_folder_path_;

	unsigned int d_iters_by_g_iter_;
	unsigned int main_iters_;

	unsigned int batch_size_;

	unsigned int z_vector_size_;
	std::string z_vector_bin_file_;
};

////////////////////////////////////////////////////////////////////////////////
struct S_InterSolverData
{
	CCifar10* cifar10_;
	CLFWFaceDatabase* faces_data;

	unsigned int z_vector_size_;
	float* z_data_;
	float* z_fix_data_;
	float* gpu_z_data_;
	float* gpu_z_fix_data_;

	unsigned int batch_size_;

	unsigned int d_iters_by_g_iter_;
	unsigned int main_iters_;

	caffe::Net<float>* net_d_;
	caffe::Net<float>* net_g_;

	unsigned int current_iter_;
	unsigned int max_iter_;

	//--------------------------------------------------------------------------
	float* gpu_ones_;
	float* gpu_zeros_;

	float* gpu_train_labels_;
	float* gpu_test_labels_;
	float* gpu_train_imgs_;
	float* gpu_test_imgs_;

	float* train_labels_;
	float* test_labels_;
	float* train_imgs_;
	float* test_imgs_;

	unsigned int count_train_;
	unsigned int count_test_;
	//--------------------------------------------------------------------------

	std::fstream* log_file_;

	std::string output_folder_path_;

	std::string solver_state_file_d_;
	std::string solver_state_file_g_;

	std::string solver_model_file_d_;
	std::string solver_model_file_g_;
};

////////////////////////////////////////////////////////////////////////////////
void usage(char *prog);

////////////////////////////////////////////////////////////////////////////////
bool isInteger(const std::string & s);
bool check_folder(const std::string& folder);

////////////////////////////////////////////////////////////////////////////////
void parse_arguments(struct S_ConfigArgs *configArgs, int argc, char **argv);

#endif /* INCLUDE_CONFIG_ARGS_HPP_ */
