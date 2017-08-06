################################################################################
# Juan Maria Gomez Lopez <juanecitorr@gmail.com>
#


################################################################################

INC_DIR := -I./include -I../my_caffe/include
SRC_DIR := ./src
BIN_DIR := ./bin
OBJ_DIR := ./obj

all: $(BIN_DIR)/wgan_debug $(BIN_DIR)/wgan_release $(BIN_DIR)/img_viewer $(BIN_DIR)/img_viewer_sequence

CXX_FLAGS_DEBUG := -g3 -O0 --std=c++14 -fPIC
CXX_FLAGS_RELEASE := -O3 --std=c++14 -fPIC

ARCH_GPU := --gpu-architecture=sm_35

NVCC_FLAGS_DEBUG := -g -O0 --std=c++11 --compiler-options -fPIC $(ARCH_GPU)
NVCC_FLAGS_RELEASE := -O3 --std=c++11 --compiler-options -fPIC $(ARCH_GPU)

INC := $(INC_DIR)

LIBS_RELEASE := -L ../my_caffe/build_r/lib -Wl,-Bdynamic -lcaffe -lcuda -lcublas -lcudart -lopencv_core -lopencv_highgui -lglog -lprotobuf -lboost_system -lopencv_imgproc -lpthread

LIBS_DEBUG := -L ../my_caffe/build_d/lib -Wl,-Bdynamic -lcaffe-d -lcuda -lcublas -lcudart -lopencv_core -lopencv_highgui -lglog -lprotobuf -lboost_system -lopencv_imgproc -lpthread

CCFILES := $(shell ls -1 ./src/*.cpp 2>/dev/null)
CUFILES := $(shell ls -1 ./src/*.cu 2>/dev/null)

OBJS_DEBUG += $(patsubst %.cpp,$(OBJ_DIR)/%.cpp.d.o,$(notdir $(CCFILES)))
OBJS_RELEASE += $(patsubst %.cpp,$(OBJ_DIR)/%.cpp.r.o,$(notdir $(CCFILES)))

OBJS_CU_DEBUG += $(patsubst %.cu,$(OBJ_DIR)/%.cu.d.o,$(notdir $(CUFILES)))
OBJS_CU_RELEASE += $(patsubst %.cu,$(OBJ_DIR)/%.cu.r.o,$(notdir $(CUFILES)))

################################################################################

$(OBJ_DIR)/%.cpp.r.o: $(SRC_DIR)/%.cpp 
	mkdir -p $(OBJ_DIR);\
	g++ $(CXX_FLAGS_RELEASE) -c $(INC) $^ -o $@

$(OBJ_DIR)/%.cpp.d.o: $(SRC_DIR)/%.cpp
	mkdir -p $(OBJ_DIR);\
	g++ $(CXX_FLAGS_DEBUG) -c $(INC) $^ -o $@

$(OBJ_DIR)/%.cu.r.o: $(SRC_DIR)/%.cu
	mkdir -p $(OBJ_DIR);\
	nvcc $(NVCC_FLAGS_RELEASE) -c $(INC) $^ -o $@

$(OBJ_DIR)/%.cu.d.o: $(SRC_DIR)/%.cu
	mkdir -p $(OBJ_DIR);\
	nvcc $(NVCC_FLAGS_DEBUG) -c $(INC) $^ -o $@

$(BIN_DIR)/wgan_release: $(OBJS_RELEASE) $(OBJS_CU_RELEASE)
	mkdir -p $(BIN_DIR);\
	g++ $(CXX_FLAGS_RELEASE) $(LIBS_RELEASE) $(INC) $^ -o $@

$(BIN_DIR)/wgan_debug: $(OBJS_DEBUG) $(OBJS_CU_DEBUG)
	mkdir -p $(BIN_DIR);\
	g++ $(CXX_FLAGS_DEBUG) $(LIBS_DEBUG) $(INC) $^ -o $@

$(BIN_DIR)/img_viewer: $(SRC_DIR)/utils/img_viewer.cpp
		g++ $(CXX_FLAGS_DEBUG) $(LIBS_DEBUG) $(INC) $^ -o $@

$(BIN_DIR)/img_viewer_sequence: $(SRC_DIR)/utils/img_viewer_sequence.cpp
		g++ $(CXX_FLAGS_DEBUG) $(LIBS_DEBUG) $(INC) $^ -o $@

################################################################################

clean:
	rm -f $(BIN_DIR)/wgan_release
	rm -f $(BIN_DIR)/wgan_debug
	rm -f $(BIN_DIR)/img_viewer
	rm -f $(BIN_DIR)/img_viewer_sequence
	rm -f $(OBJ_DIR)/*.o

################################################################################

