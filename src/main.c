#include "face_mask_detector.h"
#include "config.h"
#include "detection_engine.h"
#include "image_processing.h"

// Global application state
static app_state_t g_app_state = {0};
static volatile bool g_running = true;

// Signal handler for graceful shutdown
void signal_handler(int signum) {
    log_info("Received signal %d, initiating shutdown...", signum);
    g_running = false;
    g_app_state.running = false;
}

// Print application usage information
void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n\n", program_name);
    printf("Advanced Face Mask Detection System v%s\n\n", PROJECT_VERSION);
    printf("OPTIONS:\n");
    printf("  -c, --config FILE       Configuration file path (default: %s)\n", DEFAULT_CONFIG_FILE);
    printf("  -i, --input FILE/INDEX  Input source (file path or camera index)\n");
    printf("  -o, --output FILE       Output video file path\n");
    printf("  -m, --model FILE        Face detection model file\n");
    printf("  -M, --mask-model FILE   Mask classification model file\n");
    printf("  -t, --threshold FLOAT   Detection confidence threshold (0.0-1.0)\n");
    printf("  -n, --nms-threshold     Non-maximum suppression threshold (0.0-1.0)\n");
    printf("  -s, --size WxH          Input size for neural networks (e.g., 416x416)\n");
    printf("  -g, --gpu               Use GPU acceleration (if available)\n");
    printf("  -v, --verbose           Enable verbose logging\n");
    printf("  -q, --quiet             Disable preview window\n");
    printf("  -r, --real-time         Real-time processing mode\n");
    printf("  -S, --save-output       Save output video\n");
    printf("      --no-display        Disable GUI display\n");
    printf("      --log-file FILE     Log file path\n");
    printf("      --log-level LEVEL   Log level (debug, info, warning, error)\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -V, --version           Show version information\n\n");
    
    printf("EXAMPLES:\n");
    printf("  %s                      # Use default camera\n", program_name);
    printf("  %s -i 1                 # Use camera index 1\n", program_name);
    printf("  %s -i video.mp4         # Process video file\n", program_name);
    printf("  %s -i 0 -o output.avi   # Record from camera to file\n", program_name);
    printf("  %s -c custom.conf -g    # Use custom config with GPU\n", program_name);
    printf("\n");
}

void print_version(void) {
    printf("%s version %s\n", PROJECT_NAME, PROJECT_VERSION);
    printf("Built with OpenCV %d.%d.%d\n", CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);
    printf("Copyright (C) 2024 Face Mask Detection Project\n");
    printf("This is free software; see the source for copying conditions.\n");
}

// Parse command line arguments
int parse_arguments(int argc, char* argv[], app_config_t* config) {
    static struct option long_options[] = {
        {"config",         required_argument, 0, 'c'},
        {"input",          required_argument, 0, 'i'},
        {"output",         required_argument, 0, 'o'},
        {"model",          required_argument, 0, 'm'},
        {"mask-model",     required_argument, 0, 'M'},
        {"threshold",      required_argument, 0, 't'},
        {"nms-threshold",  required_argument, 0, 'n'},
        {"size",           required_argument, 0, 's'},
        {"gpu",            no_argument,       0, 'g'},
        {"verbose",        no_argument,       0, 'v'},
        {"quiet",          no_argument,       0, 'q'},
        {"real-time",      no_argument,       0, 'r'},
        {"save-output",    no_argument,       0, 'S'},
        {"no-display",     no_argument,       0, 1000},
        {"log-file",       required_argument, 0, 1001},
        {"log-level",      required_argument, 0, 1002},
        {"help",           no_argument,       0, 'h'},
        {"version",        no_argument,       0, 'V'},
        {0, 0, 0, 0}
    };

    int c;
    int option_index = 0;
    
    while ((c = getopt_long(argc, argv, "c:i:o:m:M:t:n:s:gvqrShV", long_options, &option_index)) != -1) {
        switch (c) {
            case 'c':
                strncpy(config->config_path, optarg, MAX_PATH_LENGTH - 1);
                break;
            case 'i':
                // Check if input is a number (camera index) or file path
                if (isdigit(optarg[0]) && strlen(optarg) <= 2) {
                    config->camera_index = atoi(optarg);
                    config->input_path[0] = '\0';
                } else {
                    strncpy(config->input_path, optarg, MAX_PATH_LENGTH - 1);
                    config->camera_index = -1;
                }
                break;
            case 'o':
                strncpy(config->output_path, optarg, MAX_PATH_LENGTH - 1);
                config->save_output = true;
                break;
            case 'm':
                strncpy(config->cascade_path, optarg, MAX_PATH_LENGTH - 1);
                break;
            case 'M':
                strncpy(config->model_path, optarg, MAX_PATH_LENGTH - 1);
                break;
            case 't':
                config->confidence_threshold = atof(optarg);
                if (config->confidence_threshold < 0.0 || config->confidence_threshold > 1.0) {
                    log_error("Confidence threshold must be between 0.0 and 1.0");
                    return FMD_ERROR_INVALID_ARGS;
                }
                break;
            case 'n':
                config->nms_threshold = atof(optarg);
                if (config->nms_threshold < 0.0 || config->nms_threshold > 1.0) {
                    log_error("NMS threshold must be between 0.0 and 1.0");
                    return FMD_ERROR_INVALID_ARGS;
                }
                break;
            case 's': {
                int w, h;
                if (sscanf(optarg, "%dx%d", &w, &h) == 2) {
                    config->input_width = w;
                    config->input_height = h;
                } else {
                    log_error("Invalid size format. Use WxH (e.g., 416x416)");
                    return FMD_ERROR_INVALID_ARGS;
                }
                break;
            }
            case 'g':
                config->use_gpu = true;
                break;
            case 'v':
                config->verbose = true;
                break;
            case 'q':
                config->show_preview = false;
                break;
            case 'r':
                config->real_time = true;
                break;
            case 'S':
                config->save_output = true;
                break;
            case 1000: // --no-display
                config->show_preview = false;
                break;
            case 'h':
                print_usage(argv[0]);
                return 1;
            case 'V':
                print_version();
                return 1;
            case '?':
                return FMD_ERROR_INVALID_ARGS;
            default:
                break;
        }
    }
    
    return FMD_SUCCESS;
}

// Initialize application state
int initialize_application(app_state_t* state, const app_config_t* config) {
    log_info("Initializing Face Mask Detection System v%s", PROJECT_VERSION);
    
    // Copy configuration
    memcpy(&state->config, config, sizeof(app_config_t));
    
    // Initialize threading primitives
    if (pthread_mutex_init(&state->frame_mutex, NULL) != 0) {
        log_error("Failed to initialize frame mutex");
        return FMD_ERROR_MEMORY_ALLOCATION;
    }
    
    if (pthread_cond_init(&state->frame_cond, NULL) != 0) {
        log_error("Failed to initialize frame condition variable");
        pthread_mutex_destroy(&state->frame_mutex);
        return FMD_ERROR_MEMORY_ALLOCATION;
    }
    
    // Load face detection cascade
    if (!state->face_cascade.load(config->cascade_path)) {
        log_error("Failed to load face cascade from: %s", config->cascade_path);
        return FMD_ERROR_MODEL_LOAD;
    }
    log_info("Loaded face detection cascade: %s", config->cascade_path);
    
    // Load mask detection model if specified (optional)
    if (strlen(config->model_path) > 0) {
        try {
            state->mask_net = cv::dnn::readNet(config->model_path);
            if (state->mask_net.empty()) {
                log_warning("Failed to load mask detection model: %s. Using heuristic-based detection.", config->model_path);
            } else {
                // Set backend and target
                if (config->use_gpu) {
                    state->mask_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                    state->mask_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                    log_info("Using GPU acceleration for mask detection");
                } else {
                    state->mask_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                    state->mask_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                    log_info("Using CPU for mask detection");
                }
                
                log_info("Loaded mask detection model: %s", config->model_path);
            }
        } catch (const cv::Exception& e) {
            log_warning("OpenCV exception while loading model: %s. Using heuristic-based detection.", e.what());
            // Clear the network so it will fall back to heuristic detection
            state->mask_net = cv::dnn::Net();
        }
    } else {
        log_info("No mask detection model specified. Using heuristic-based detection.");
    }
    
    // Initialize camera or video file
    if (strlen(config->input_path) > 0) {
        // Video file input
        if (!state->cap.open(config->input_path)) {
            log_error("Failed to open video file: %s", config->input_path);
            return FMD_ERROR_CAMERA_INIT;
        }
        log_info("Opened video file: %s", config->input_path);
    } else {
        // Camera input
        if (!state->cap.open(config->camera_index)) {
            log_error("Failed to open camera with index: %d", config->camera_index);
            return FMD_ERROR_CAMERA_INIT;
        }
        log_info("Opened camera with index: %d", config->camera_index);
        
        // Set camera properties for better performance
        state->cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        state->cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        state->cap.set(cv::CAP_PROP_FPS, 30);
    }
    
    // Initialize video writer if output is requested
    if (config->save_output && strlen(config->output_path) > 0) {
        int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
        double fps = state->cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;
        
        cv::Size frame_size((int)state->cap.get(cv::CAP_PROP_FRAME_WIDTH),
                           (int)state->cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        
        if (!state->writer.open(config->output_path, fourcc, fps, frame_size)) {
            log_warning("Failed to initialize video writer for: %s", config->output_path);
        } else {
            log_info("Initialized video writer: %s", config->output_path);
        }
    }
    
    state->running = true;
    state->detection_count = 0;
    state->frame_count = 0;
    state->fps = 0.0;
    
    log_info("Application initialization completed successfully");
    return FMD_SUCCESS;
}

// Cleanup application resources
void cleanup_application(app_state_t* state) {
    log_info("Cleaning up application resources...");
    
    state->running = false;
    
    // Release video capture and writer
    if (state->cap.isOpened()) {
        state->cap.release();
    }
    
    if (state->writer.isOpened()) {
        state->writer.release();
    }
    
    // Cleanup threading primitives
    pthread_cond_destroy(&state->frame_cond);
    pthread_mutex_destroy(&state->frame_mutex);
    
    // Cleanup OpenCV windows
    cv::destroyAllWindows();
    
    log_info("Application cleanup completed");
}

// Main processing loop
int run_detection_loop(app_state_t* state) {
    cv::Mat frame;
    double start_time, end_time;
    double fps_timer = get_current_time();
    int frame_count = 0;
    
    log_info("Starting detection loop...");
    
    while (state->running && g_running) {
        start_time = get_current_time();
        
        // Capture frame
        if (!state->cap.read(frame)) {
            if (strlen(state->config.input_path) > 0) {
                // End of video file
                log_info("Reached end of video file");
                break;
            } else {
                log_error("Failed to capture frame from camera");
                continue;
            }
        }
        
        if (frame.empty()) {
            continue;
        }
        
        // Detect faces
        int face_count = detect_faces(state, frame, state->detections, MAX_FACES);
        state->detection_count = face_count;
        
        // Draw detections on frame
        if (face_count > 0) {
            draw_detections(frame, state->detections, face_count);
        }
        
        // Display frame
        if (state->config.show_preview) {
            cv::imshow("Face Mask Detection", frame);
            
            int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q') { // ESC or 'q' to quit
                log_info("User requested quit");
                break;
            }
            
            // Handle other key inputs
            handle_key_input(state, key);
        }
        
        // Save frame if recording
        if (state->writer.isOpened()) {
            state->writer.write(frame);
        }
        
        // Calculate FPS
        end_time = get_current_time();
        frame_count++;
        state->frame_count++;
        
        if (end_time - fps_timer >= 1.0) {
            state->fps = frame_count / (end_time - fps_timer);
            if (state->config.verbose) {
                log_info("FPS: %.2f, Faces detected: %d", state->fps, face_count);
            }
            frame_count = 0;
            fps_timer = end_time;
        }
        
        // Real-time processing delay
        if (state->config.real_time) {
            double processing_time = end_time - start_time;
            double target_frame_time = 1.0 / 30.0; // 30 FPS
            if (processing_time < target_frame_time) {
                usleep((target_frame_time - processing_time) * 1000000);
            }
        }
    }
    
    log_info("Detection loop completed. Processed %llu frames", state->frame_count);
    return FMD_SUCCESS;
}

// Main function
int main(int argc, char* argv[]) {
    app_config_t config;
    int result;
    
    // Set default configuration
    set_default_config(&config);
    
    // Parse command line arguments
    result = parse_arguments(argc, argv, &config);
    if (result != FMD_SUCCESS) {
        if (result == 1) return 0; // Help or version displayed
        return result;
    }
    
    // Initialize logging (basic console logging for now)
    log_info("Starting %s v%s", PROJECT_NAME, PROJECT_VERSION);
    
    // Load configuration file if specified
    if (strlen(config.config_path) > 0) {
        if (load_config(&config, config.config_path) != FMD_SUCCESS) {
            log_warning("Failed to load config file: %s", config.config_path);
        }
    }
    
    // Print configuration if verbose
    if (config.verbose) {
        print_config(&config);
    }
    
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Initialize application
    result = initialize_application(&g_app_state, &config);
    if (result != FMD_SUCCESS) {
        log_error("Failed to initialize application: %s", error_to_string((fmd_error_t)result));
        cleanup_application(&g_app_state);
        return result;
    }
    
    // Run main processing loop
    result = run_detection_loop(&g_app_state);
    
    // Cleanup and exit
    cleanup_application(&g_app_state);
    
    if (result == FMD_SUCCESS) {
        log_info("Application completed successfully");
        return 0;
    } else {
        log_error("Application exited with error: %s", error_to_string((fmd_error_t)result));
        return result;
    }
}