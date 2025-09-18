#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include "face_mask_detector.h"

#ifdef __cplusplus
extern "C" {
#endif

// Image format enumeration
typedef enum {
    IMAGE_FORMAT_RGB = 0,
    IMAGE_FORMAT_BGR = 1,
    IMAGE_FORMAT_GRAY = 2,
    IMAGE_FORMAT_HSV = 3,
    IMAGE_FORMAT_YUV = 4
} image_format_t;

// Image enhancement parameters
typedef struct {
    float brightness;
    float contrast;
    float gamma;
    float saturation;
    bool histogram_equalization;
    bool noise_reduction;
} enhancement_params_t;

// ROI (Region of Interest) structure
typedef struct {
    int x, y, width, height;
    bool valid;
} roi_t;

// Image statistics
typedef struct {
    double mean;
    double std_dev;
    double min_val;
    double max_val;
    int histogram[256];
} image_stats_t;

// Core image processing functions
int load_image(const char* path, cv::Mat& image);
int save_image(const char* path, const cv::Mat& image);
int resize_image(const cv::Mat& input, cv::Mat& output, int width, int height, int interpolation);
int convert_color_space(const cv::Mat& input, cv::Mat& output, image_format_t from, image_format_t to);

// Image enhancement functions
int enhance_image(const cv::Mat& input, cv::Mat& output, const enhancement_params_t* params);
int adjust_brightness_contrast(const cv::Mat& input, cv::Mat& output, float brightness, float contrast);
int apply_gamma_correction(const cv::Mat& input, cv::Mat& output, float gamma);
int reduce_noise(const cv::Mat& input, cv::Mat& output);
int apply_histogram_equalization(const cv::Mat& input, cv::Mat& output);

// Preprocessing functions for ML models
int preprocess_for_detection(const cv::Mat& input, cv::Mat& output, int target_size, bool normalize);
int create_blob_from_image(const cv::Mat& image, cv::Mat& blob, double scale_factor, 
                          const cv::Size& size, const cv::Scalar& mean, bool swap_rb);

// ROI and cropping functions
int extract_roi(const cv::Mat& input, cv::Mat& output, const roi_t* roi);
int crop_face_region(const cv::Mat& input, cv::Mat& output, const face_detection_t* face, 
                     int padding, int target_size);

// Image analysis functions
int calculate_image_stats(const cv::Mat& image, image_stats_t* stats);
int compute_histogram(const cv::Mat& image, int* histogram, int bins);
double calculate_image_quality(const cv::Mat& image);
double calculate_blur_score(const cv::Mat& image);

// Filtering and smoothing
int apply_gaussian_blur(const cv::Mat& input, cv::Mat& output, int kernel_size, double sigma);
int apply_median_filter(const cv::Mat& input, cv::Mat& output, int kernel_size);
int apply_bilateral_filter(const cv::Mat& input, cv::Mat& output, int d, double sigma_color, double sigma_space);

// Edge detection and features
int detect_edges(const cv::Mat& input, cv::Mat& output, double threshold1, double threshold2);
int detect_corners(const cv::Mat& input, std::vector<cv::Point2f>& corners, int max_corners, double quality, double min_distance);

// Utility functions
void set_default_enhancement_params(enhancement_params_t* params);
const char* image_format_to_string(image_format_t format);
bool is_valid_roi(const roi_t* roi, int image_width, int image_height);
roi_t create_roi(int x, int y, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_PROCESSING_H