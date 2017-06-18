################################################################################
# Juan Maria Gomez Lopez <juanecitorr@gmail.com>
#


################################################################################

INC_DIR := ./include
SRC_DIR := ./src
BIN_DIR := ./bin
OBJ_DIR := ./obj

all: $(BIN_DIR)/wgan_debug $(BIN_DIR)/wgan_release

CXX_FLAGS_DEBUG := -g3 -O0

CXX_FLAGS_RELEASE := -O3

INC := -I$(INC_DIR)

LIBS := -lopencv_core -lcaffe -lopencv_highgui -lglog -lprotobuf -lboost_system -lopencv_imgproc

CCFILES := $(shell ls -1 ./src/*.cpp 2>/dev/null)

OBJS_DEBUG += $(patsubst %.cpp,$(OBJ_DIR)/%.cpp.d.o,$(notdir $(CCFILES)))
OBJS_RELEASE += $(patsubst %.cpp,$(OBJ_DIR)/%.cpp.r.o,$(notdir $(CCFILES)))

################################################################################

$(OBJ_DIR)/%.cpp.r.o: $(SRC_DIR)/%.cpp 
	mkdir -p $(OBJ_DIR);\
	g++ $(CXX_FLAGS_RELEASE) -c $(INC) $^ -o $@

$(OBJ_DIR)/%.cpp.d.o: $(SRC_DIR)/%.cpp
	mkdir -p $(OBJ_DIR);\
	g++ $(CXX_FLAGS_DEBUG) -c $(INC) $^ -o $@

$(BIN_DIR)/wgan_release: $(OBJS_RELEASE)
	mkdir -p $(BIN_DIR);\
	g++ $(CXX_FLAGS_RELEASE) $(LIBS) $(INC) $^ -o $@

$(BIN_DIR)/wgan_debug: $(OBJS_DEBUG)
	mkdir -p $(BIN_DIR);\
	g++ $(CXX_FLAGS_DEBUG) $(LIBS) $(INC) $^ -o $@

################################################################################

clean:
	rm -f $(BIN_DIR)/wgan_release
	rm -f $(BIN_DIR)/wgan_debug
	rm -f $(OBJ_DIR)/*.o

################################################################################

