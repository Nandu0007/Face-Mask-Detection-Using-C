#include "image_processing.h"
#include "face_mask_detector.h"

// Load image from file
int load_image(const char* path, cv::Mat& image) {
    if (!path || strlen(path) == 0) {
        log_error("Invalid image path provided");
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        image = cv::imread(path, cv::IMREAD_COLOR);
        if (image.empty()) {
            log_error("Failed to load image: %s", path);
            return FMD_ERROR_FILE_NOT_FOUND;
        }
        
        log_debug("Loaded image: %s (Size: %dx%d)", path, image.cols, image.rows);
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while loading image: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Save image to file
int save_image(const char* path, const cv::Mat& image) {
    if (!path || strlen(path) == 0) {
        log_error("Invalid output path provided");
        return FMD_ERROR_INVALID_ARGS;
    }
    
    if (image.empty()) {
        log_error("Cannot save empty image");
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        if (!cv::imwrite(path, image)) {
            log_error("Failed to save image: %s", path);
            return FMD_ERROR_PROCESSING;
        }
        
        log_debug("Saved image: %s", path);
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while saving image: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Resize image
int resize_image(const cv::Mat& input, cv::Mat& output, int width, int height, int interpolation) {
    if (input.empty()) {
        log_error("Cannot resize empty image");
        return FMD_ERROR_INVALID_ARGS;
    }
    
    if (width <= 0 || height <= 0) {
        log_error("Invalid target dimensions: %dx%d", width, height);
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        cv::resize(input, output, cv::Size(width, height), 0, 0, interpolation);
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while resizing image: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Convert color space
int convert_color_space(const cv::Mat& input, cv::Mat& output, image_format_t from, image_format_t to) {
    if (input.empty()) {
        log_error("Cannot convert empty image");
        return FMD_ERROR_INVALID_ARGS;
    }
    
    if (from == to) {
        output = input.clone();
        return FMD_SUCCESS;
    }
    
    try {
        int cv_code = -1;
        
        // Determine OpenCV color conversion code
        if (from == IMAGE_FORMAT_BGR && to == IMAGE_FORMAT_RGB) {
            cv_code = cv::COLOR_BGR2RGB;
        } else if (from == IMAGE_FORMAT_RGB && to == IMAGE_FORMAT_BGR) {
            cv_code = cv::COLOR_RGB2BGR;
        } else if (from == IMAGE_FORMAT_BGR && to == IMAGE_FORMAT_GRAY) {
            cv_code = cv::COLOR_BGR2GRAY;
        } else if (from == IMAGE_FORMAT_RGB && to == IMAGE_FORMAT_GRAY) {
            cv_code = cv::COLOR_RGB2GRAY;
        } else if (from == IMAGE_FORMAT_BGR && to == IMAGE_FORMAT_HSV) {
            cv_code = cv::COLOR_BGR2HSV;
        } else if (from == IMAGE_FORMAT_RGB && to == IMAGE_FORMAT_HSV) {
            cv_code = cv::COLOR_RGB2HSV;
        } else if (from == IMAGE_FORMAT_HSV && to == IMAGE_FORMAT_BGR) {
            cv_code = cv::COLOR_HSV2BGR;
        } else if (from == IMAGE_FORMAT_HSV && to == IMAGE_FORMAT_RGB) {
            cv_code = cv::COLOR_HSV2RGB;
        } else if (from == IMAGE_FORMAT_BGR && to == IMAGE_FORMAT_YUV) {
            cv_code = cv::COLOR_BGR2YUV;
        } else if (from == IMAGE_FORMAT_RGB && to == IMAGE_FORMAT_YUV) {
            cv_code = cv::COLOR_RGB2YUV;
        } else {
            log_error("Unsupported color space conversion: %s to %s", 
                     image_format_to_string(from), image_format_to_string(to));
            return FMD_ERROR_INVALID_ARGS;
        }
        
        cv::cvtColor(input, output, cv_code);
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while converting color space: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Enhance image with multiple parameters
int enhance_image(const cv::Mat& input, cv::Mat& output, const enhancement_params_t* params) {
    if (input.empty() || !params) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    cv::Mat working = input.clone();
    
    try {
        // Apply brightness and contrast
        if (params->brightness != 0.0f || params->contrast != 1.0f) {
            working.convertTo(working, -1, params->contrast, params->brightness);
        }
        
        // Apply gamma correction
        if (params->gamma != 1.0f) {
            apply_gamma_correction(working, working, params->gamma);
        }
        
        // Apply histogram equalization
        if (params->histogram_equalization) {
            apply_histogram_equalization(working, working);
        }
        
        // Apply noise reduction
        if (params->noise_reduction) {
            reduce_noise(working, working);
        }
        
        output = working;
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while enhancing image: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Adjust brightness and contrast
int adjust_brightness_contrast(const cv::Mat& input, cv::Mat& output, float brightness, float contrast) {
    if (input.empty()) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        input.convertTo(output, -1, contrast, brightness);
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while adjusting brightness/contrast: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Apply gamma correction
int apply_gamma_correction(const cv::Mat& input, cv::Mat& output, float gamma) {
    if (input.empty()) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        // Create lookup table for gamma correction
        cv::Mat lookup_table(1, 256, CV_8U);
        uchar* p = lookup_table.ptr();
        for (int i = 0; i < 256; ++i) {
            p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
        }
        
        cv::LUT(input, lookup_table, output);
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while applying gamma correction: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Reduce noise
int reduce_noise(const cv::Mat& input, cv::Mat& output) {
    if (input.empty()) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        // Use bilateral filter for noise reduction while preserving edges
        cv::bilateralFilter(input, output, 9, 75, 75);
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while reducing noise: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Apply histogram equalization
int apply_histogram_equalization(const cv::Mat& input, cv::Mat& output) {
    if (input.empty()) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        if (input.channels() == 1) {
            // Grayscale image
            cv::equalizeHist(input, output);
        } else {
            // Color image - convert to YUV, equalize Y channel, convert back
            cv::Mat yuv, channels[3];
            cv::cvtColor(input, yuv, cv::COLOR_BGR2YUV);
            cv::split(yuv, channels);
            cv::equalizeHist(channels[0], channels[0]);
            cv::merge(channels, 3, yuv);
            cv::cvtColor(yuv, output, cv::COLOR_YUV2BGR);
        }
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while equalizing histogram: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Preprocess for detection models
int preprocess_for_detection(const cv::Mat& input, cv::Mat& output, int target_size, bool normalize) {
    if (input.empty() || target_size <= 0) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        cv::Mat resized;
        cv::resize(input, resized, cv::Size(target_size, target_size));
        
        if (normalize) {
            resized.convertTo(output, CV_32F, 1.0/255.0);
        } else {
            output = resized;
        }
        
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while preprocessing for detection: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Create blob from image for DNN
int create_blob_from_image(const cv::Mat& image, cv::Mat& blob, double scale_factor, 
                          const cv::Size& size, const cv::Scalar& mean, bool swap_rb) {
    if (image.empty()) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        blob = cv::dnn::blobFromImage(image, scale_factor, size, mean, swap_rb, false, CV_32F);
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while creating blob: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Extract region of interest
int extract_roi(const cv::Mat& input, cv::Mat& output, const roi_t* roi) {
    if (input.empty() || !roi || !roi->valid) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    if (!is_valid_roi(roi, input.cols, input.rows)) {
        log_error("ROI is outside image boundaries");
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        cv::Rect rect(roi->x, roi->y, roi->width, roi->height);
        output = input(rect).clone();
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while extracting ROI: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Crop face region with padding
int crop_face_region(const cv::Mat& input, cv::Mat& output, const face_detection_t* face, 
                     int padding, int target_size) {
    if (input.empty() || !face) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        // Calculate padded rectangle
        int x = std::max(0, face->x - padding);
        int y = std::max(0, face->y - padding);
        int width = std::min(input.cols - x, face->width + 2 * padding);
        int height = std::min(input.rows - y, face->height + 2 * padding);
        
        cv::Rect face_rect(x, y, width, height);
        cv::Mat cropped = input(face_rect).clone();
        
        if (target_size > 0) {
            cv::resize(cropped, output, cv::Size(target_size, target_size));
        } else {
            output = cropped;
        }
        
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while cropping face region: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Calculate image statistics
int calculate_image_stats(const cv::Mat& image, image_stats_t* stats) {
    if (image.empty() || !stats) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        cv::Mat gray;
        if (image.channels() > 1) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }
        
        cv::Scalar mean, stddev;
        cv::meanStdDev(gray, mean, stddev);
        
        double min_val, max_val;
        cv::minMaxLoc(gray, &min_val, &max_val);
        
        stats->mean = mean[0];
        stats->std_dev = stddev[0];
        stats->min_val = min_val;
        stats->max_val = max_val;
        
        // Calculate histogram
        compute_histogram(gray, stats->histogram, 256);
        
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while calculating image stats: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Compute histogram
int compute_histogram(const cv::Mat& image, int* histogram, int bins) {
    if (image.empty() || !histogram || bins <= 0) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        cv::Mat hist;
        int histSize[] = {bins};
        float range[] = {0, 256};
        const float* histRange[] = {range};
        
        cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, histSize, histRange);
        
        for (int i = 0; i < bins; i++) {
            histogram[i] = (int)hist.at<float>(i);
        }
        
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while computing histogram: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Calculate image quality score
double calculate_image_quality(const cv::Mat& image) {
    if (image.empty()) {
        return 0.0;
    }
    
    try {
        cv::Mat gray;
        if (image.channels() > 1) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }
        
        // Calculate Laplacian variance (focus measure)
        cv::Mat laplacian;
        cv::Laplacian(gray, laplacian, CV_64F);
        
        cv::Scalar mean, stddev;
        cv::meanStdDev(laplacian, mean, stddev);
        
        double variance = stddev[0] * stddev[0];
        
        // Normalize to 0-1 range (empirically determined threshold)
        return std::min(1.0, variance / 1000.0);
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while calculating image quality: %s", e.what());
        return 0.0;
    }
}

// Calculate blur score
double calculate_blur_score(const cv::Mat& image) {
    if (image.empty()) {
        return 0.0;
    }
    
    try {
        cv::Mat gray;
        if (image.channels() > 1) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image;
        }
        
        cv::Mat laplacian;
        cv::Laplacian(gray, laplacian, CV_64F);
        
        cv::Scalar mean, stddev;
        cv::meanStdDev(laplacian, mean, stddev);
        
        return stddev[0] * stddev[0];
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while calculating blur score: %s", e.what());
        return 0.0;
    }
}

// Apply Gaussian blur
int apply_gaussian_blur(const cv::Mat& input, cv::Mat& output, int kernel_size, double sigma) {
    if (input.empty() || kernel_size <= 0 || kernel_size % 2 == 0) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        cv::GaussianBlur(input, output, cv::Size(kernel_size, kernel_size), sigma);
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while applying Gaussian blur: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Apply median filter
int apply_median_filter(const cv::Mat& input, cv::Mat& output, int kernel_size) {
    if (input.empty() || kernel_size <= 0 || kernel_size % 2 == 0) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        cv::medianBlur(input, output, kernel_size);
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while applying median filter: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Apply bilateral filter
int apply_bilateral_filter(const cv::Mat& input, cv::Mat& output, int d, double sigma_color, double sigma_space) {
    if (input.empty()) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        cv::bilateralFilter(input, output, d, sigma_color, sigma_space);
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while applying bilateral filter: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Detect edges
int detect_edges(const cv::Mat& input, cv::Mat& output, double threshold1, double threshold2) {
    if (input.empty()) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        cv::Mat gray;
        if (input.channels() > 1) {
            cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = input;
        }
        
        cv::Canny(gray, output, threshold1, threshold2);
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception while detecting edges: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Utility functions
void set_default_enhancement_params(enhancement_params_t* params) {
    if (!params) return;
    
    params->brightness = 0.0f;
    params->contrast = 1.0f;
    params->gamma = 1.0f;
    params->saturation = 1.0f;
    params->histogram_equalization = false;
    params->noise_reduction = false;
}

const char* image_format_to_string(image_format_t format) {
    switch (format) {
        case IMAGE_FORMAT_RGB: return "RGB";
        case IMAGE_FORMAT_BGR: return "BGR";
        case IMAGE_FORMAT_GRAY: return "GRAY";
        case IMAGE_FORMAT_HSV: return "HSV";
        case IMAGE_FORMAT_YUV: return "YUV";
        default: return "UNKNOWN";
    }
}

bool is_valid_roi(const roi_t* roi, int image_width, int image_height) {
    if (!roi || !roi->valid) return false;
    
    return (roi->x >= 0 && roi->y >= 0 && 
            roi->x + roi->width <= image_width && 
            roi->y + roi->height <= image_height &&
            roi->width > 0 && roi->height > 0);
}

roi_t create_roi(int x, int y, int width, int height) {
    roi_t roi;
    roi.x = x;
    roi.y = y;
    roi.width = width;
    roi.height = height;
    roi.valid = (width > 0 && height > 0);
    return roi;
}