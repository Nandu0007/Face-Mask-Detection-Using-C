#ifndef FACE_MASK_DETECTOR_H
#define FACE_MASK_DETECTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <getopt.h>
#include <signal.h>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn.hpp>

// Project version
#define PROJECT_VERSION "1.0.0"
#define PROJECT_NAME "Face Mask Detector"

// Configuration constants
#define MAX_PATH_LENGTH 256
#define MAX_STRING_LENGTH 128
#define MAX_FACES 20
#define DEFAULT_CAMERA_INDEX 0
#define DEFAULT_CONFIDENCE_THRESHOLD 0.5
#define DEFAULT_NMS_THRESHOLD 0.4
#define DEFAULT_INPUT_SIZE 416

// Error codes
typedef enum {
    FMD_SUCCESS = 0,
    FMD_ERROR_INVALID_ARGS = -1,
    FMD_ERROR_FILE_NOT_FOUND = -2,
    FMD_ERROR_MEMORY_ALLOCATION = -3,
    FMD_ERROR_OPENCV_INIT = -4,
    FMD_ERROR_MODEL_LOAD = -5,
    FMD_ERROR_CAMERA_INIT = -6,
    FMD_ERROR_PROCESSING = -7
} fmd_error_t;

// Detection result
typedef enum {
    MASK_STATUS_UNKNOWN = 0,
    MASK_STATUS_WITH_MASK = 1,
    MASK_STATUS_WITHOUT_MASK = 2,
    MASK_STATUS_INCORRECT_MASK = 3
} mask_status_t;

// Face detection structure
typedef struct {
    int x, y, width, height;
    float confidence;
    mask_status_t mask_status;
    float mask_confidence;
    // Enhanced temporal smoothing for stable detection
    int mask_history[10];  // Last 10 detections (larger window)
    int history_index;
    int history_count;
    mask_status_t stable_status;  // Current stable status
    int stable_count;            // How long we've been stable
} face_detection_t;

// Application configuration
typedef struct {
    char model_path[MAX_PATH_LENGTH];
    char config_path[MAX_PATH_LENGTH];
    char cascade_path[MAX_PATH_LENGTH];
    char input_path[MAX_PATH_LENGTH];
    char output_path[MAX_PATH_LENGTH];
    int camera_index;
    float confidence_threshold;
    float nms_threshold;
    int input_width;
    int input_height;
    bool use_gpu;
    bool save_output;
    bool show_preview;
    bool verbose;
    bool real_time;
} app_config_t;

// Application state
typedef struct {
    app_config_t config;
    cv::CascadeClassifier face_cascade;
    cv::dnn::Net mask_net;
    cv::VideoCapture cap;
    cv::VideoWriter writer;
    bool running;
    pthread_mutex_t frame_mutex;
    pthread_cond_t frame_cond;
    cv::Mat current_frame;
    face_detection_t detections[MAX_FACES];
    int detection_count;
    uint64_t frame_count;
    double fps;
} app_state_t;

// Function prototypes
extern "C" {

// Core functions
int fmd_init(app_state_t* state, const app_config_t* config);
void fmd_cleanup(app_state_t* state);
int fmd_run(app_state_t* state);

// Configuration functions
int load_config(app_config_t* config, const char* config_file);
void set_default_config(app_config_t* config);
void print_config(const app_config_t* config);

// Detection functions
int detect_faces(app_state_t* state, const cv::Mat& frame, face_detection_t* faces, int max_faces);
int classify_mask(app_state_t* state, const cv::Mat& frame, const face_detection_t* face, mask_status_t* status, float* confidence);
mask_status_t classify_mask_simple_reliable(const cv::Mat& frame, const face_detection_t* face);
mask_status_t apply_temporal_smoothing(face_detection_t* face, mask_status_t current_status);

// Image processing functions
int preprocess_frame(const cv::Mat& input, cv::Mat& output, int target_width, int target_height);
void draw_detections(cv::Mat& frame, const face_detection_t* faces, int count);
const char* mask_status_to_string(mask_status_t status);

// UI functions
int init_ui(app_state_t* state);
void cleanup_ui(app_state_t* state);
int display_frame(app_state_t* state, const cv::Mat& frame);
int handle_key_input(app_state_t* state, int key);

// Utility functions
double get_current_time(void);
void print_usage(const char* program_name);
void print_version(void);
const char* error_to_string(fmd_error_t error);

// Logging functions
void log_info(const char* format, ...);
void log_warning(const char* format, ...);
void log_error(const char* format, ...);
void log_debug(const char* format, ...);

}

#endif // FACE_MASK_DETECTOR_H