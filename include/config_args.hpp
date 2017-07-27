/*
 * config_args.hpp
 *
 *  Created on: 27 jul. 2017
 *      Author: juanitov
 */

#ifndef INCLUDE_CONFIG_ARGS_HPP_
#define INCLUDE_CONFIG_ARGS_HPP_

#include <string>

enum E_CONFIG_ARGS_OPTS
{
	OPT_LOG,
	OPT_PRE_TRAIN_FILE,
	OPT_WGAN_D_SOLVER_MODEL_FILE,
	OPT_WGAN_G_SOLVER_MODEL_FILE,
	OPT_WGAN_D_SOLVER_STATE_FILE,
	OPT_WGAN_G_SOLVER_STATE_FILE,
	OPT_DATA_SOURCE_FOLDER_PATH
};

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

	std::string data_source_folder_path_;
};



#endif /* INCLUDE_CONFIG_ARGS_HPP_ */
