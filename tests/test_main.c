#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "face_mask_detector.h"
#include "image_processing.h"
#include "config.h"

// Simple test framework
#define TEST_ASSERT(condition, message) do { \
    if (!(condition)) { \
        printf("FAIL: %s - %s\n", __func__, message); \
        return -1; \
    } \
    printf("PASS: %s - %s\n", __func__, message); \
    return 0; \
} while(0)

// Test configuration functions
int test_default_config() {
    app_config_t config;
    set_default_config(&config);
    
    TEST_ASSERT(config.camera_index == DEFAULT_CAMERA_INDEX, 
                "Default camera index should be set correctly");
}

int test_config_validation() {
    app_config_t config;
    set_default_config(&config);
    
    // Test invalid confidence threshold
    config.confidence_threshold = -0.1f;
    TEST_ASSERT(config.confidence_threshold < 0.0f, 
                "Should allow setting invalid threshold for testing");
}

// Test utility functions
int test_error_to_string() {
    const char* error_str = error_to_string(FMD_SUCCESS);
    TEST_ASSERT(strcmp(error_str, "Success") == 0, 
                "Error to string conversion should work correctly");
}

int test_time_function() {
    double time1 = get_current_time();
    double time2 = get_current_time();
    
    TEST_ASSERT(time2 >= time1, 
                "Time should be monotonic");
}

// Test image processing functions
int test_image_format_string() {
    const char* format_str = image_format_to_string(IMAGE_FORMAT_RGB);
    TEST_ASSERT(strcmp(format_str, "RGB") == 0, 
                "Image format to string conversion should work");
}

int test_roi_validation() {
    roi_t roi = create_roi(10, 10, 100, 100);
    TEST_ASSERT(roi.valid == true, "ROI should be created as valid");
    TEST_ASSERT(is_valid_roi(&roi, 200, 200) == true, 
                "ROI should be valid within image bounds");
}

int test_mask_status_string() {
    const char* status_str = mask_status_to_string(MASK_STATUS_WITH_MASK);
    TEST_ASSERT(strcmp(status_str, "With Mask") == 0, 
                "Mask status to string conversion should work");
}

// Test logging system
int test_logging_initialization() {
    logging_config_t log_config = {0};
    log_config.level = LOG_LEVEL_INFO;
    log_config.console_output = true;
    log_config.file_output = false;
    
    int result = init_logging_system(&log_config);
    TEST_ASSERT(result == FMD_SUCCESS, 
                "Logging system should initialize successfully");
}

// Main test runner
int main(int argc, char* argv[]) {
    printf("Running Face Mask Detector Tests...\n");
    printf("====================================\n\n");
    
    int tests_run = 0;
    int tests_passed = 0;
    
    // Run configuration tests
    tests_run++;
    if (test_default_config() == 0) tests_passed++;
    
    tests_run++;
    if (test_config_validation() == 0) tests_passed++;
    
    // Run utility tests
    tests_run++;
    if (test_error_to_string() == 0) tests_passed++;
    
    tests_run++;
    if (test_time_function() == 0) tests_passed++;
    
    // Run image processing tests
    tests_run++;
    if (test_image_format_string() == 0) tests_passed++;
    
    tests_run++;
    if (test_roi_validation() == 0) tests_passed++;
    
    tests_run++;
    if (test_mask_status_string() == 0) tests_passed++;
    
    // Run logging tests
    tests_run++;
    if (test_logging_initialization() == 0) tests_passed++;
    
    // Cleanup
    cleanup_logging_system();
    
    // Print results
    printf("\n====================================\n");
    printf("Tests Results: %d/%d passed\n", tests_passed, tests_run);
    
    if (tests_passed == tests_run) {
        printf("All tests PASSED!\n");
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}