#include "face_mask_detector.h"
#include "config.h"
#include <sys/time.h>
#include <stdarg.h>

// Global logging state
static log_level_t g_log_level = LOG_LEVEL_INFO;
static FILE* g_log_file = NULL;
static bool g_console_logging = true;

// Set default configuration
void set_default_config(app_config_t* config) {
    if (!config) return;
    
    memset(config, 0, sizeof(app_config_t));
    
    // Set default paths
    strncpy(config->cascade_path, DEFAULT_CASCADE_FILE, MAX_PATH_LENGTH - 1);
    strncpy(config->model_path, DEFAULT_MASK_MODEL_FILE, MAX_PATH_LENGTH - 1);
    strncpy(config->config_path, DEFAULT_CONFIG_FILE, MAX_PATH_LENGTH - 1);
    
    // Set default values
    config->camera_index = DEFAULT_CAMERA_INDEX;
    config->confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD;
    config->nms_threshold = DEFAULT_NMS_THRESHOLD;
    config->input_width = DEFAULT_INPUT_SIZE;
    config->input_height = DEFAULT_INPUT_SIZE;
    
    // Set default flags
    config->use_gpu = false;
    config->save_output = false;
    config->show_preview = true;
    config->verbose = false;
    config->real_time = true;
}

// Load configuration from file
int load_config(app_config_t* config, const char* config_file) {
    if (!config || !config_file) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    FILE* file = fopen(config_file, "r");
    if (!file) {
        log_warning("Could not open config file: %s", config_file);
        return FMD_ERROR_FILE_NOT_FOUND;
    }
    
    char line[256];
    char key[64], value[128];
    int line_number = 0;
    
    while (fgets(line, sizeof(line), file)) {
        line_number++;
        
        // Skip empty lines and comments
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        
        // Parse key=value pairs
        if (sscanf(line, "%63[^=]=%127[^\r\n]", key, value) == 2) {
            // Trim whitespace
            char* key_trimmed = key;
            while (*key_trimmed == ' ' || *key_trimmed == '\t') key_trimmed++;
            char* key_end = key_trimmed + strlen(key_trimmed) - 1;
            while (key_end > key_trimmed && (*key_end == ' ' || *key_end == '\t')) key_end--;
            *(key_end + 1) = '\0';
            
            char* value_trimmed = value;
            while (*value_trimmed == ' ' || *value_trimmed == '\t') value_trimmed++;
            char* value_end = value_trimmed + strlen(value_trimmed) - 1;
            while (value_end > value_trimmed && (*value_end == ' ' || *value_end == '\t')) value_end--;
            *(value_end + 1) = '\0';
            
            // Apply configuration
            if (strcmp(key_trimmed, "cascade_path") == 0) {
                strncpy(config->cascade_path, value_trimmed, MAX_PATH_LENGTH - 1);
            } else if (strcmp(key_trimmed, "model_path") == 0) {
                strncpy(config->model_path, value_trimmed, MAX_PATH_LENGTH - 1);
            } else if (strcmp(key_trimmed, "camera_index") == 0) {
                config->camera_index = atoi(value_trimmed);
            } else if (strcmp(key_trimmed, "confidence_threshold") == 0) {
                config->confidence_threshold = atof(value_trimmed);
            } else if (strcmp(key_trimmed, "nms_threshold") == 0) {
                config->nms_threshold = atof(value_trimmed);
            } else if (strcmp(key_trimmed, "input_width") == 0) {
                config->input_width = atoi(value_trimmed);
            } else if (strcmp(key_trimmed, "input_height") == 0) {
                config->input_height = atoi(value_trimmed);
            } else if (strcmp(key_trimmed, "use_gpu") == 0) {
                config->use_gpu = (strcmp(value_trimmed, "true") == 0 || strcmp(value_trimmed, "1") == 0);
            } else if (strcmp(key_trimmed, "show_preview") == 0) {
                config->show_preview = (strcmp(value_trimmed, "true") == 0 || strcmp(value_trimmed, "1") == 0);
            } else if (strcmp(key_trimmed, "verbose") == 0) {
                config->verbose = (strcmp(value_trimmed, "true") == 0 || strcmp(value_trimmed, "1") == 0);
            } else {
                log_warning("Unknown configuration key '%s' at line %d", key_trimmed, line_number);
            }
        }
    }
    
    fclose(file);
    log_info("Loaded configuration from: %s", config_file);
    return FMD_SUCCESS;
}

// Print current configuration
void print_config(const app_config_t* config) {
    if (!config) return;
    
    printf("\n=== Face Mask Detection Configuration ===\n");
    printf("Cascade Path:          %s\n", config->cascade_path);
    printf("Model Path:            %s\n", config->model_path);
    printf("Config Path:           %s\n", config->config_path);
    printf("Input Path:            %s\n", config->input_path);
    printf("Output Path:           %s\n", config->output_path);
    printf("Camera Index:          %d\n", config->camera_index);
    printf("Confidence Threshold:  %.3f\n", config->confidence_threshold);
    printf("NMS Threshold:         %.3f\n", config->nms_threshold);
    printf("Input Size:            %dx%d\n", config->input_width, config->input_height);
    printf("Use GPU:               %s\n", config->use_gpu ? "Yes" : "No");
    printf("Save Output:           %s\n", config->save_output ? "Yes" : "No");
    printf("Show Preview:          %s\n", config->show_preview ? "Yes" : "No");
    printf("Verbose:               %s\n", config->verbose ? "Yes" : "No");
    printf("Real-time Mode:        %s\n", config->real_time ? "Yes" : "No");
    printf("==========================================\n\n");
}

// Get current time in seconds (with microsecond precision)
double get_current_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Convert error code to string
const char* error_to_string(fmd_error_t error) {
    switch (error) {
        case FMD_SUCCESS: return "Success";
        case FMD_ERROR_INVALID_ARGS: return "Invalid arguments";
        case FMD_ERROR_FILE_NOT_FOUND: return "File not found";
        case FMD_ERROR_MEMORY_ALLOCATION: return "Memory allocation failed";
        case FMD_ERROR_OPENCV_INIT: return "OpenCV initialization failed";
        case FMD_ERROR_MODEL_LOAD: return "Model loading failed";
        case FMD_ERROR_CAMERA_INIT: return "Camera initialization failed";
        case FMD_ERROR_PROCESSING: return "Processing error";
        default: return "Unknown error";
    }
}

// Initialize logging system
int init_logging_system(const logging_config_t* config) {
    if (!config) {
        g_log_level = LOG_LEVEL_INFO;
        g_console_logging = true;
        return FMD_SUCCESS;
    }
    
    g_log_level = config->level;
    g_console_logging = config->console_output;
    
    if (config->file_output && strlen(config->log_file) > 0) {
        g_log_file = fopen(config->log_file, "a");
        if (!g_log_file) {
            fprintf(stderr, "Warning: Could not open log file: %s\n", config->log_file);
        }
    }
    
    return FMD_SUCCESS;
}

// Cleanup logging system
void cleanup_logging_system(void) {
    if (g_log_file) {
        fclose(g_log_file);
        g_log_file = NULL;
    }
}

// Internal logging function
static void log_message(log_level_t level, const char* format, va_list args) {
    if (level < g_log_level) return;
    
    // Get current time
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);
    
    // Level string
    const char* level_str;
    switch (level) {
        case LOG_LEVEL_DEBUG: level_str = "DEBUG"; break;
        case LOG_LEVEL_INFO: level_str = "INFO"; break;
        case LOG_LEVEL_WARNING: level_str = "WARN"; break;
        case LOG_LEVEL_ERROR: level_str = "ERROR"; break;
        case LOG_LEVEL_FATAL: level_str = "FATAL"; break;
        default: level_str = "UNKNOWN"; break;
    }
    
    // Format message
    char message[1024];
    vsnprintf(message, sizeof(message), format, args);
    
    // Log to console
    if (g_console_logging) {
        FILE* output = (level >= LOG_LEVEL_ERROR) ? stderr : stdout;
        fprintf(output, "[%s] %s: %s\n", timestamp, level_str, message);
        fflush(output);
    }
    
    // Log to file
    if (g_log_file) {
        fprintf(g_log_file, "[%s] %s: %s\n", timestamp, level_str, message);
        fflush(g_log_file);
    }
}

// Logging functions
void log_debug(const char* format, ...) {
    va_list args;
    va_start(args, format);
    log_message(LOG_LEVEL_DEBUG, format, args);
    va_end(args);
}

void log_info(const char* format, ...) {
    va_list args;
    va_start(args, format);
    log_message(LOG_LEVEL_INFO, format, args);
    va_end(args);
}

void log_warning(const char* format, ...) {
    va_list args;
    va_start(args, format);
    log_message(LOG_LEVEL_WARNING, format, args);
    va_end(args);
}

void log_error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    log_message(LOG_LEVEL_ERROR, format, args);
    va_end(args);
}

// UI functions (basic implementations)
int init_ui(app_state_t* state) {
    if (!state) return FMD_ERROR_INVALID_ARGS;
    
    if (state->config.show_preview) {
        cv::namedWindow("Face Mask Detection", cv::WINDOW_AUTOSIZE);
        log_info("Initialized UI window");
    }
    
    return FMD_SUCCESS;
}

void cleanup_ui(app_state_t* state) {
    if (!state) return;
    
    cv::destroyAllWindows();
    log_info("Cleaned up UI");
}

int display_frame(app_state_t* state, const cv::Mat& frame) {
    if (!state || frame.empty()) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    if (state->config.show_preview) {
        cv::imshow("Face Mask Detection", frame);
    }
    
    return FMD_SUCCESS;
}

// Create default configuration file
int create_default_config_file(const char* config_path) {
    if (!config_path) return FMD_ERROR_INVALID_ARGS;
    
    FILE* file = fopen(config_path, "w");
    if (!file) {
        log_error("Could not create config file: %s", config_path);
        return FMD_ERROR_FILE_NOT_FOUND;
    }
    
    fprintf(file, "# Face Mask Detection Configuration File\n");
    fprintf(file, "# Generated automatically\n\n");
    
    fprintf(file, "[Models]\n");
    fprintf(file, "cascade_path = %s\n", DEFAULT_CASCADE_FILE);
    fprintf(file, "model_path = %s\n", DEFAULT_MASK_MODEL_FILE);
    fprintf(file, "\n");
    
    fprintf(file, "[Detection]\n");
    fprintf(file, "confidence_threshold = %.3f\n", DEFAULT_CONFIDENCE_THRESHOLD);
    fprintf(file, "nms_threshold = %.3f\n", DEFAULT_NMS_THRESHOLD);
    fprintf(file, "input_width = %d\n", DEFAULT_INPUT_SIZE);
    fprintf(file, "input_height = %d\n", DEFAULT_INPUT_SIZE);
    fprintf(file, "\n");
    
    fprintf(file, "[General]\n");
    fprintf(file, "camera_index = %d\n", DEFAULT_CAMERA_INDEX);
    fprintf(file, "use_gpu = false\n");
    fprintf(file, "show_preview = true\n");
    fprintf(file, "verbose = false\n");
    fprintf(file, "\n");
    
    fprintf(file, "[Logging]\n");
    fprintf(file, "log_level = info\n");
    fprintf(file, "log_file = %s\n", DEFAULT_LOG_FILE);
    fprintf(file, "console_output = true\n");
    fprintf(file, "file_output = false\n");
    
    fclose(file);
    log_info("Created default configuration file: %s", config_path);
    return FMD_SUCCESS;
}

// Missing function declaration
mask_status_t classify_mask_heuristic(const cv::Mat& frame, const face_detection_t* face);

// Image resize function implementation
int resize_image(const cv::Mat& input, cv::Mat& output, int width, int height, int interpolation) {
    if (input.empty()) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    try {
        cv::resize(input, output, cv::Size(width, height), 0, 0, interpolation);
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception in resize_image: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Preprocessor frame wrapper (to satisfy the function call in main.c)
int preprocess_frame(const cv::Mat& input, cv::Mat& output, int target_width, int target_height) {
    return resize_image(input, output, target_width, target_height, cv::INTER_LINEAR);
}
