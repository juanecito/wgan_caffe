#####################################################




######################################################

SRC_DIR := ./src
BIN_DIR := ./bin

all: $(BIN_DIR)/wgan_debug $(BIN_DIR)/wgan_release

CXX_FLAGS_DEBUG := -g3 -O0

CXX_FLAGS_RELEASE := -O3

INC := -I./include

LIBS := -lopencv_core -lcaffe -lopencv_highgui -lglog -lprotobuf -lboost_system -lopencv_imgproc

$(BIN_DIR)/wgan_release: $(SRC_DIR)/main.cpp $(SRC_DIR)/CCifar10.cpp
	mkdir -p $(BIN_DIR);\
	g++ $(CXX_FLAGS_RELEASE) $(LIBS) $(INC) $^ -o $@

$(BIN_DIR)/wgan_debug: $(SRC_DIR)/main.cpp $(SRC_DIR)/CCifar10.cpp
	mkdir -p $(BIN_DIR);\
	g++ $(CXX_FLAGS_DEBUG) $(LIBS) $(INC) $^ -o $@

clean:
	rm -f $(BIN_DIR)/wgan_release
	rm -f $(BIN_DIR)/wgan_debug