#ifndef DETECTION_ENGINE_H
#define DETECTION_ENGINE_H

#include "face_mask_detector.h"
#include "image_processing.h"

#ifdef __cplusplus
extern "C" {
#endif

// Detection model types
typedef enum {
    MODEL_TYPE_HAAR_CASCADE = 0,
    MODEL_TYPE_DNN_CAFFE = 1,
    MODEL_TYPE_DNN_TENSORFLOW = 2,
    MODEL_TYPE_DNN_DARKNET = 3,
    MODEL_TYPE_DNN_ONNX = 4
} model_type_t;

// Detection backend
typedef enum {
    DETECTION_BACKEND_OPENCV = 0,
    DETECTION_BACKEND_CUDA = 1,
    DETECTION_BACKEND_OPENCL = 2,
    DETECTION_BACKEND_CPU = 3
} detection_backend_t;

// Model configuration
typedef struct {
    model_type_t type;
    detection_backend_t backend;
    char model_path[MAX_PATH_LENGTH];
    char config_path[MAX_PATH_LENGTH];
    char classes_path[MAX_PATH_LENGTH];
    int input_width;
    int input_height;
    float scale_factor;
    cv::Scalar mean;
    bool swap_rb;
    float confidence_threshold;
    float nms_threshold;
} model_config_t;

// Detection parameters
typedef struct {
    double scale_factor;
    int min_neighbors;
    int min_size_width;
    int min_size_height;
    int max_size_width;
    int max_size_height;
    bool do_canny_pruning;
} detection_params_t;

// Performance metrics
typedef struct {
    double detection_time_ms;
    double preprocessing_time_ms;
    double inference_time_ms;
    double postprocessing_time_ms;
    int faces_detected;
    int faces_with_mask;
    int faces_without_mask;
    double average_confidence;
} detection_metrics_t;

// Detection engine state
typedef struct {
    cv::CascadeClassifier face_classifier;
    cv::dnn::Net mask_network;
    model_config_t face_model_config;
    model_config_t mask_model_config;
    detection_params_t face_detection_params;
    detection_backend_t current_backend;
    bool initialized;
    detection_metrics_t metrics;
} detection_engine_t;

// Core detection functions
int init_detection_engine(detection_engine_t* engine, const model_config_t* face_config, const model_config_t* mask_config);
void cleanup_detection_engine(detection_engine_t* engine);
int detect_faces_in_frame(detection_engine_t* engine, const cv::Mat& frame, face_detection_t* faces, int max_faces, int* count);
int classify_mask_status(detection_engine_t* engine, const cv::Mat& frame, const face_detection_t* face, 
                        mask_status_t* status, float* confidence);

// Model loading and configuration
int load_face_detection_model(detection_engine_t* engine, const model_config_t* config);
int load_mask_classification_model(detection_engine_t* engine, const model_config_t* config);
int set_detection_backend(detection_engine_t* engine, detection_backend_t backend);
int validate_model_config(const model_config_t* config);

// Haar Cascade specific functions
int detect_faces_haar(detection_engine_t* engine, const cv::Mat& frame, face_detection_t* faces, int max_faces, int* count);
int optimize_haar_parameters(detection_engine_t* engine, const cv::Mat& sample_frame);

// DNN specific functions
int detect_faces_dnn(detection_engine_t* engine, const cv::Mat& frame, face_detection_t* faces, int max_faces, int* count);
int run_mask_classification_dnn(detection_engine_t* engine, const cv::Mat& face_roi, mask_status_t* status, float* confidence);
int preprocess_for_dnn(const cv::Mat& input, cv::Mat& blob, const model_config_t* config);
int postprocess_detections(const cv::Mat& output, face_detection_t* faces, int max_faces, int* count, 
                          float confidence_threshold, float nms_threshold);

// Batch processing functions
int detect_faces_batch(detection_engine_t* engine, const cv::Mat* frames, int frame_count, 
                      face_detection_t** all_faces, int* face_counts, int max_faces_per_frame);
int process_video_stream(detection_engine_t* engine, cv::VideoCapture& capture, 
                        void (*callback)(const cv::Mat&, const face_detection_t*, int, void*), void* user_data);

// Performance optimization
int optimize_detection_parameters(detection_engine_t* engine, const cv::Mat* sample_frames, int sample_count);
void reset_performance_metrics(detection_engine_t* engine);
void update_performance_metrics(detection_engine_t* engine, double detection_time, int faces_detected, 
                               int masked, int unmasked);

// Utility and helper functions
void set_default_model_config(model_config_t* config, model_type_t type);
void set_default_detection_params(detection_params_t* params);
const char* model_type_to_string(model_type_t type);
const char* backend_to_string(detection_backend_t backend);
bool is_model_file_valid(const char* path);
int get_optimal_input_size(int image_width, int image_height, int max_size);

// Non-maximum suppression
int apply_nms(face_detection_t* faces, int count, float threshold);
float calculate_iou(const face_detection_t* face1, const face_detection_t* face2);

// Tracking and temporal consistency
typedef struct {
    int track_id;
    face_detection_t last_detection;
    mask_status_t stable_mask_status;
    int consecutive_detections;
    double last_update_time;
    bool active;
} face_track_t;

int init_face_tracking(face_track_t* tracks, int max_tracks);
int update_face_tracks(face_track_t* tracks, int max_tracks, const face_detection_t* detections, 
                      int detection_count, double current_time);
int get_stable_mask_status(const face_track_t* track);

#ifdef __cplusplus
}
#endif

#endif // DETECTION_ENGINE_H