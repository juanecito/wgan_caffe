#####################################################




######################################################

SRC_DIR := ./src
BIN_DIR := ./bin

all: $(BIN_DIR)/wgan

LIBS := -lopencv_core -lcaffe -lopencv_highgui -lglog -lprotobuf -lboost_system -lopencv_imgproc

$(BIN_DIR)/wgan: $(SRC_DIR)/main.cpp
	mkdir $(BIN_DIR);\
	g++ -g3 -O0 $(LIBS) $< -o $@
	
clean:
	rm -f $(BIN_DIR)/wgan