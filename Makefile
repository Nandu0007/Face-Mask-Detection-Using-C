# Face Mask Detection - Advanced C Project
# Makefile for building the application

# Project settings
PROJECT_NAME = face_mask_detector
VERSION = 1.0.0

# Directories
SRC_DIR = src
INCLUDE_DIR = include
LIB_DIR = lib
BUILD_DIR = build
BIN_DIR = bin
TEST_DIR = tests

# Compiler settings
CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -std=c11 -O2 -g
CXXFLAGS = -Wall -Wextra -std=c++11 -O2 -g
INCLUDES = -I$(INCLUDE_DIR) -I/usr/local/include -I/opt/homebrew/include -I/opt/homebrew/Cellar/opencv/4.12.0_11/include/opencv4
LIBS = -L/usr/local/lib -L/opt/homebrew/lib -L/opt/homebrew/Cellar/opencv/4.12.0_11/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_objdetect -lopencv_videoio -lopencv_video -lopencv_dnn -lm -lpthread

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
TARGET = $(BIN_DIR)/$(PROJECT_NAME)

# Test files
TEST_SOURCES = $(wildcard $(TEST_DIR)/*.c)
TEST_OBJECTS = $(TEST_SOURCES:$(TEST_DIR)/%.c=$(BUILD_DIR)/test_%.o)
TEST_TARGET = $(BIN_DIR)/test_runner

# Default target
.PHONY: all clean install uninstall test help

all: $(TARGET)

# Create directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build main target
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(OBJECTS) -o $@ $(LIBS)
	@echo "Build complete: $(TARGET)"

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Build tests
test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJECTS) $(filter-out $(BUILD_DIR)/main.o, $(OBJECTS)) | $(BIN_DIR)
	$(CXX) $^ -o $@ $(LIBS)

$(BUILD_DIR)/test_%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Development targets
debug: CXXFLAGS += -DDEBUG -g3 -O0
debug: $(TARGET)

release: CXXFLAGS += -DNDEBUG -O3
release: $(TARGET)

# Install target
install: $(TARGET)
	sudo cp $(TARGET) /usr/local/bin/
	sudo mkdir -p /usr/local/share/$(PROJECT_NAME)
	sudo cp -r models /usr/local/share/$(PROJECT_NAME)/ 2>/dev/null || true
	sudo cp -r config /usr/local/share/$(PROJECT_NAME)/ 2>/dev/null || true

# Uninstall target
uninstall:
	sudo rm -f /usr/local/bin/$(PROJECT_NAME)
	sudo rm -rf /usr/local/share/$(PROJECT_NAME)

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "Clean complete"

# Package for distribution
dist: clean
	tar -czf $(PROJECT_NAME)-$(VERSION).tar.gz --exclude='.git' .

# Format code (requires clang-format)
format:
	find $(SRC_DIR) $(INCLUDE_DIR) -name "*.c" -o -name "*.h" | xargs clang-format -i

# Static analysis (requires cppcheck)
analyze:
	cppcheck --enable=all --std=c11 $(SRC_DIR) $(INCLUDE_DIR)

# Help target
help:
	@echo "Available targets:"
	@echo "  all      - Build the project (default)"
	@echo "  debug    - Build with debug flags"
	@echo "  release  - Build optimized release version"
	@echo "  test     - Build and run tests"
	@echo "  clean    - Remove build files"
	@echo "  install  - Install to system"
	@echo "  uninstall- Remove from system"
	@echo "  format   - Format source code"
	@echo "  analyze  - Run static analysis"
	@echo "  dist     - Create distribution package"
	@echo "  help     - Show this help"

# Dependencies
-include $(OBJECTS:.o=.d)