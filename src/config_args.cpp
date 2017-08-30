/*
 * config_args.cpp
 *
 *  Created on: 29 jul. 2017
 *      Author: juanitov
 */

#include <getopt.h>
#include <dirent.h>
#include <sys/stat.h>

#include "config_args.hpp"

////////////////////////////////////////////////////////////////////////////////
void usage(char *prog)
{
  fprintf(stderr, "usage: %s [-help] [-log arg] [-train-file arg] [-M arg]\n", prog);
  exit(1);
}

////////////////////////////////////////////////////////////////////////////////
bool isInteger(const std::string & s)
{
   if(s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+'))) return false ;

   char* p = nullptr;
   strtol(s.c_str(), &p, 10) ;

   return (*p == '\0');
}

////////////////////////////////////////////////////////////////////////////////
bool check_folder(const std::string& folder)
{
	struct stat st;
	if (stat(folder.c_str(), &st) != 0)
	{
		mode_t mode = 0777;
		/* Directory does not exist, I'm goint to create it */
		if (mkdir(folder.c_str(), mode) != 0)
		{
			return false;
		}
	}
	else
	{
		if (!S_ISDIR(st.st_mode))
		{
			return false;
		}
	}
	return true;
}

////////////////////////////////////////////////////////////////////////////////
void parse_arguments(struct S_ConfigArgs *configArgs, int argc, char **argv)
{
	configArgs->solver_d_model_.clear();
	configArgs->solver_g_model_.clear();
	configArgs->solver_d_state_.clear();
	configArgs->solver_g_state_.clear();
	configArgs->data_source_folder_path_.clear();
	configArgs->run_wgan_ = 0;
	configArgs->run_cifar10_training_ = 0;
	configArgs->test_cifar10_ = 0;
	configArgs->batch_size_ = 0;
	configArgs->d_iters_by_g_iter_ = 0;
	configArgs->main_iters_ = 0;
	configArgs->z_vector_size_ = 0;
	configArgs->z_vector_bin_file_.clear();
	configArgs->logarg_ = {0};

	int c, option_index, err;
	static const char short_options[] = "W:P:C:f:n";
	static const struct option long_options[] = {
		{"help", no_argument, 0, 'h'},

        {"run-wgan", no_argument, &(configArgs->run_wgan_), 1},
        {"cifar10-train",   no_argument, &(configArgs->run_cifar10_training_), 1},
        {"cifar10-test",   no_argument, &(configArgs->test_cifar10_), 1},
		{"log",      required_argument, 0, OPT_LOG},
		{"dataset", required_argument, 0, OPT_DATASET},

		{"data-src-path", required_argument, 0, OPT_DATA_SOURCE_FOLDER_PATH},
		{"output-path", required_argument, 0, OPT_OUTPUT_FOLDER_PATH},

		{"train-file", required_argument, 0, OPT_PRE_TRAIN_FILE},
		{"solver-d-model", required_argument, 0, OPT_WGAN_D_SOLVER_MODEL_FILE},
		{"solver-g-model", required_argument, 0, OPT_WGAN_G_SOLVER_MODEL_FILE},
		{"solver-d-state", required_argument, 0, OPT_WGAN_D_SOLVER_STATE_FILE},
		{"solver-g-state", required_argument, 0, OPT_WGAN_G_SOLVER_STATE_FILE},

		{"d-iters-by-g-iter", required_argument, 0, OPT_D_ITERS_BY_G_ITER},
		{"main-iter", required_argument, 0, OPT_MAIN_ITERS},

		{"z-vector-bin-file", required_argument, 0, OPT_Z_VECTOR_BIN_FILE},
		{"z-vector-size", required_argument, 0, OPT_Z_VECTOR_SIZE},
		{"batch-size", required_argument, 0, OPT_BATCH_SIZE},
		{0, 0, 0, 0}
	};

	while ((c = getopt_long(argc, argv, short_options, long_options,
					&option_index)) != -1)
	{
		switch (c)
		{
		case OPT_LOG: configArgs->logarg_ = optarg; break;
		case OPT_DATASET: configArgs->dataset_ = optarg; break; // LFW_faces, Cifar10,
		case OPT_PRE_TRAIN_FILE: configArgs->pretrain_file_name_ = optarg; break;
		case OPT_WGAN_D_SOLVER_MODEL_FILE: configArgs->solver_d_model_ = optarg; break;
		case OPT_WGAN_G_SOLVER_MODEL_FILE: configArgs->solver_g_model_ = optarg; break;
		case OPT_WGAN_D_SOLVER_STATE_FILE: configArgs->solver_d_state_ = optarg; break;
		case OPT_WGAN_G_SOLVER_STATE_FILE: configArgs->solver_g_state_= optarg; break;
		case OPT_DATA_SOURCE_FOLDER_PATH: configArgs->data_source_folder_path_= optarg; break;
		case OPT_OUTPUT_FOLDER_PATH: configArgs->output_folder_path_ = optarg; break;
		case OPT_Z_VECTOR_BIN_FILE: configArgs->z_vector_bin_file_ = optarg; break;

		case OPT_D_ITERS_BY_G_ITER:
			if (isInteger(optarg)){ configArgs->d_iters_by_g_iter_ = abs(atoi(optarg));}
			else { usage(argv[0]); abort();}
			break;

		case OPT_MAIN_ITERS:
			if (isInteger(optarg)){ configArgs->main_iters_ = abs(atoi(optarg));}
			else { usage(argv[0]); abort();}
			break;

		case OPT_BATCH_SIZE:
			if (isInteger(optarg)) {configArgs->batch_size_ = abs(atoi(optarg));}
			else{ usage(argv[0]); abort();}
			break;

		case OPT_Z_VECTOR_SIZE:
			if (isInteger(optarg)){ configArgs->z_vector_size_ = abs(atoi(optarg));}
			else {usage(argv[0]); abort();}
			break;

		case 'W':
			// TODO:
			break;

		case 'C':
			// TODO:
			break;

		case 'n':
			// TODO:
			break;

		case 'h':
			usage(argv[0]);
			exit(EXIT_SUCCESS);
		default:
			usage(argv[0]);
			abort();
		}
	}
}
