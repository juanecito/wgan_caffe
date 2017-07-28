#!/bin/bash
################################################################################

WGAN_BIN_FILE=./bin/wgan_release

# Learning rate in A models is base_lr: 0.0001
SOLVER_D_MODEL_A=./models/solver_d_lr_A.prototxt
SOLVER_G_MODEL_A=./models/solver_g_lr_A.prototxt

# Learning rate in B models is base_lr: 0.00001
SOLVER_D_MODEL_B=./models/solver_d_lr_B.prototxt
SOLVER_G_MODEL_B=./models/solver_g_lr_B.prototxt

#SOLVER_D_STATE_B=./wgan_d_iter_7525.solverstate
#SOLVER_G_STATE_B=./wgan_g_iter_301.solverstate

SOLVER_D_STATE_B=./wgan_d_iter_15025.solverstate
SOLVER_G_STATE_B=./wgan_g_iter_601.solverstate

################################################################################

. ./wgan_exports

#${WGAN_BIN_FILE} --run-wgan --solver-d-model ${SOLVER_D_MODEL_A} --solver-g-model ${SOLVER_G_MODEL_A} --solver-d-state c --solver-g-state d

${WGAN_BIN_FILE} --run-wgan --solver-d-model ${SOLVER_D_MODEL_B} --solver-g-model ${SOLVER_G_MODEL_B} --solver-d-state ${SOLVER_D_STATE_B} --solver-g-state ${SOLVER_G_STATE_B}

################################################################################

exit 0
