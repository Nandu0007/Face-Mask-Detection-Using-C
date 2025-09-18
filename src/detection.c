#include "face_mask_detector.h"
#include "image_processing.h"

// Apply temporal smoothing to reduce detection noise
mask_status_t apply_temporal_smoothing(face_detection_t* face, mask_status_t current_status) {
    if (!face) return current_status;
    
    // Initialize history buffer on first call
    if (face->history_count == 0) {
        for (int i = 0; i < 10; i++) {
            face->mask_history[i] = current_status;
        }
        face->history_index = 0;
        face->history_count = 1;
        face->stable_status = current_status;
        face->stable_count = 1;
        return current_status;
    }
    
    // Status locking to prevent flickering
    static mask_status_t current_locked_status = MASK_STATUS_UNKNOWN;
    static int lock_frames_remaining = 0;
    static int same_result_count = 0;
    static mask_status_t previous_result = MASK_STATUS_UNKNOWN;
    
    // Count consecutive identical results
    if (current_status == previous_result) {
        same_result_count++;
    } else {
        same_result_count = 1;
        previous_result = current_status;
    }
    
    // Handle status locking logic
    if (current_locked_status != MASK_STATUS_UNKNOWN) {
        lock_frames_remaining--;
        
        if (lock_frames_remaining > 0) {
            // Check for special case: faster mask removal
            if (current_locked_status == MASK_STATUS_WITH_MASK && 
                current_status == MASK_STATUS_WITHOUT_MASK && 
                same_result_count >= 8) {
                // Quick transition when removing mask
                current_locked_status = current_status;
                lock_frames_remaining = 60;
                return current_locked_status;
            }
            return current_locked_status;
        } else {
            // Lock has expired, check if we should change
            if (same_result_count >= 12 && current_status != current_locked_status) {
                int lock_duration = (current_status == MASK_STATUS_WITH_MASK) ? 90 : 60;
                current_locked_status = current_status;
                lock_frames_remaining = lock_duration;
                return current_locked_status;
            } else if (same_result_count < 12) {
                // Extend the lock a bit more
                lock_frames_remaining = 20;
                return current_locked_status;
            }
        }
    } else {
        // No active lock, establish one if we have consistent results
        if (same_result_count >= 5) {
            int lock_duration = (current_status == MASK_STATUS_WITH_MASK) ? 90 : 60;
            current_locked_status = current_status;
            lock_frames_remaining = lock_duration;
            return current_locked_status;
        }
    }
    
    // Debug output every second or so
    static int debug_frame_count = 0;
    if (++debug_frame_count % 30 == 0) {
        log_info("Detection status: %s (count: %d)", 
                current_status == MASK_STATUS_WITH_MASK ? "MASK" : 
                current_status == MASK_STATUS_WITHOUT_MASK ? "NO-MASK" : "UNKNOWN", 
                same_result_count);
        if (current_locked_status != MASK_STATUS_UNKNOWN) {
            log_info("Locked to: %s, frames left: %d", 
                    current_locked_status == MASK_STATUS_WITH_MASK ? "MASK" : "NO-MASK", 
                    lock_frames_remaining);
        }
    }
    
    // Fallback: if no lock is active and not enough consistency, use previous stable status
    // or default to current input if we don't have a better option
    if (face->stable_status != MASK_STATUS_UNKNOWN) {
        return face->stable_status;
    }
    
    return current_status;
}

// Detect faces using Haar cascade
int detect_faces(app_state_t* state, const cv::Mat& frame, face_detection_t* faces, int max_faces) {
    if (!state || frame.empty() || !faces || max_faces <= 0) {
        return 0;
    }
    
    try {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        // Apply histogram equalization for better detection
        cv::equalizeHist(gray, gray);
        
        std::vector<cv::Rect> face_rects;
        
        // Primary detection - optimized for glasses
        state->face_cascade.detectMultiScale(
            gray,
            face_rects,
            1.05,   // fine-grained scaling
            2,      // less strict neighbor requirement
            cv::CASCADE_SCALE_IMAGE,
            cv::Size(24, 24), 
            cv::Size(300, 300)
        );
        
        // Try backup cascade if nothing found
        if (face_rects.empty()) {
            cv::CascadeClassifier backup_cascade;
            if (backup_cascade.load("models/haarcascade_frontalface_default.xml")) {
                backup_cascade.detectMultiScale(
                    gray,
                    face_rects,
                    1.1,
                    3,
                    0,
                    cv::Size(30, 30)
                );
            }
        }
        
        // Last resort - try LBP based detection
        if (face_rects.empty()) {
            cv::CascadeClassifier lbp_detector;
            if (lbp_detector.load("models/lbpcascade_frontalface_improved.xml")) {
                lbp_detector.detectMultiScale(
                    gray,
                    face_rects,
                    1.1,
                    2,
                    0,
                    cv::Size(20, 20)
                );
            }
        }
        
        int count = std::min((int)face_rects.size(), max_faces);
        
        // Debug face detection with cascade info
        static int face_debug_counter = 0;
        static int last_face_count = -1;
        
        if (++face_debug_counter % 30 == 0 || (int)face_rects.size() != last_face_count) {
            log_info("*** FACE DETECTION DEBUG ***");
            log_info("Detected %d faces (max=%d)", (int)face_rects.size(), max_faces);
            
            if (face_rects.size() == 0) {
                log_info("NO FACES DETECTED - Tried multiple cascades");
                log_info("TROUBLESHOOTING:");
                log_info("- Remove glasses temporarily to test");
                log_info("- Ensure good lighting");
                log_info("- Face camera directly");
                log_info("- Move closer/farther from camera");
            } else {
                log_info("SUCCESS: Face detection working");
                for (size_t i = 0; i < face_rects.size() && i < 3; i++) {
                    log_info("Face %zu: x=%d y=%d w=%d h=%d", i, face_rects[i].x, face_rects[i].y, face_rects[i].width, face_rects[i].height);
                }
            }
            log_info("**************************");
            last_face_count = (int)face_rects.size();
        }
        
        for (int i = 0; i < count; i++) {
            faces[i].x = face_rects[i].x;
            faces[i].y = face_rects[i].y;
            faces[i].width = face_rects[i].width;
            faces[i].height = face_rects[i].height;
            faces[i].confidence = 1.0f; // Haar cascade doesn't provide confidence
            
            // Classify mask status for each face
            mask_status_t raw_mask_status = MASK_STATUS_UNKNOWN;
            float mask_confidence = 0.0f;
            
            if (!state->mask_net.empty()) {
                classify_mask(state, frame, &faces[i], &raw_mask_status, &mask_confidence);
            } else {
                // Simple reliable classification when no ML model is available
                raw_mask_status = classify_mask_simple_reliable(frame, &faces[i]);
                mask_confidence = 0.80f; // Good confidence for simple reliable method
            }
            
            // Apply temporal smoothing to prevent flickering
            mask_status_t smooth_status = apply_temporal_smoothing(&faces[i], raw_mask_status);
            
            faces[i].mask_status = smooth_status;
            faces[i].mask_confidence = mask_confidence;
        }
        
        return count;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception in face detection: %s", e.what());
        return 0;
    }
}

// Classify mask status using ML model
int classify_mask(app_state_t* state, const cv::Mat& frame, const face_detection_t* face, 
                  mask_status_t* status, float* confidence) {
    if (!state || !face || !status || !confidence) {
        return FMD_ERROR_INVALID_ARGS;
    }
    
    *status = MASK_STATUS_UNKNOWN;
    *confidence = 0.0f;
    
    if (state->mask_net.empty()) {
        log_warning("Mask classification model not loaded");
        return FMD_ERROR_MODEL_LOAD;
    }
    
    try {
        // Extract face region
        cv::Mat face_roi;
        int result = crop_face_region(frame, face_roi, face, 10, 224);
        if (result != FMD_SUCCESS) {
            return result;
        }
        
        // Create blob for DNN
        cv::Mat blob;
        cv::dnn::blobFromImage(face_roi, blob, 1.0/255.0, cv::Size(224, 224), 
                              cv::Scalar(0.485, 0.456, 0.406), true, false, CV_32F);
        
        // Set input to the network
        state->mask_net.setInput(blob);
        
        // Run inference
        cv::Mat output = state->mask_net.forward();
        
        // Parse output (assuming binary classification: mask/no-mask)
        if (output.total() >= 2) {
            float* data = (float*)output.data;
            float no_mask_conf = data[0];
            float mask_conf = data[1];
            
            if (mask_conf > no_mask_conf) {
                *status = MASK_STATUS_WITH_MASK;
                *confidence = mask_conf;
            } else {
                *status = MASK_STATUS_WITHOUT_MASK;
                *confidence = no_mask_conf;
            }
        } else {
            log_error("Unexpected output format from mask classification model");
            return FMD_ERROR_PROCESSING;
        }
        
        return FMD_SUCCESS;
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception in mask classification: %s", e.what());
        return FMD_ERROR_PROCESSING;
    }
}

// Improved heuristic-based mask classification (fallback when no ML model)
mask_status_t classify_mask_heuristic(const cv::Mat& frame, const face_detection_t* face) {
    if (frame.empty() || !face) {
        return MASK_STATUS_UNKNOWN;
    }
    
    try {
        // Extract face region with bounds checking
        cv::Rect face_rect(face->x, face->y, face->width, face->height);
        
        // Ensure face rect is within frame bounds
        face_rect &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (face_rect.width < 10 || face_rect.height < 10) {
            return MASK_STATUS_UNKNOWN;
        }
        
        cv::Mat face_roi = frame(face_rect);
        
        // Convert to different color spaces for analysis
        cv::Mat hsv, gray;
        cv::cvtColor(face_roi, hsv, cv::COLOR_BGR2HSV);
        cv::cvtColor(face_roi, gray, cv::COLOR_BGR2GRAY);
        
        // Define regions for analysis - focus more on mouth area for proper mask detection
        int mouth_y = face_roi.rows * 0.65;  // Lower part of face (mainly mouth area)
        int mouth_height = face_roi.rows * 0.25;  // Smaller region focused on mouth
        
        cv::Rect mouth_nose_rect(face_roi.cols * 0.2, mouth_y, 
                                face_roi.cols * 0.6, mouth_height);
        mouth_nose_rect &= cv::Rect(0, 0, face_roi.cols, face_roi.rows);
        
        cv::Mat mouth_nose_hsv = hsv(mouth_nose_rect);
        cv::Mat mouth_nose_gray = gray(mouth_nose_rect);
        
        // Calculate various features
        cv::Scalar mean_hsv = cv::mean(mouth_nose_hsv);
        cv::Scalar mean_gray = cv::mean(mouth_nose_gray);
        
        // Calculate standard deviation for texture analysis
        cv::Scalar stddev_gray;
        cv::meanStdDev(mouth_nose_gray, cv::Scalar(), stddev_gray);
        
        // Feature extraction
        double hue = mean_hsv[0];
        double saturation = mean_hsv[1];
        double value = mean_hsv[2];
        double brightness = mean_gray[0];
        double texture = stddev_gray[0];
        
        // Additional analysis for proper mask detection
        // Check for mixed regions (skin + mask) which indicates proper mask wearing
        cv::Mat skin_mask, non_skin_mask;
        // More accurate skin detection range - wider hue range, better saturation/value
        cv::inRange(mouth_nose_hsv, cv::Scalar(0, 30, 50), cv::Scalar(30, 150, 255), skin_mask);
        cv::bitwise_not(skin_mask, non_skin_mask);
        
        int skin_pixels = cv::countNonZero(skin_mask);
        int non_skin_pixels = cv::countNonZero(non_skin_mask);
        int total_pixels = mouth_nose_rect.width * mouth_nose_rect.height;
        
        double skin_ratio = (double)skin_pixels / total_pixels;
        double non_skin_ratio = (double)non_skin_pixels / total_pixels;
        
        // Enhanced mask detection with proper mask analysis
        int mask_score = 0;
        int no_mask_score = 0;
        
        // Balanced mask detection based on skin vs non-skin ratios
        if (non_skin_ratio > 0.6) {
            // Very high non-skin = definitely wearing mask
            mask_score += 6;
        } else if (non_skin_ratio > 0.4) {
            // High non-skin = probably wearing mask
            mask_score += 4;
        } else if (non_skin_ratio > 0.25) {
            // Some non-skin = might be wearing mask
            mask_score += 2;
        } 
        
        if (skin_ratio > 0.7) {
            // High skin ratio = probably no mask
            no_mask_score += 3;
        }
        if (skin_ratio > 0.85) {
            // Very high skin ratio = definitely no mask
            no_mask_score += 3;
        }
        
        // Skin color detection (human skin has specific HSV ranges)
        bool looks_like_skin = false;
        if ((hue > 5 && hue < 20) || (hue > 165 && hue < 175)) {  // More restrictive skin hue range
            if (saturation > 50 && saturation < 110 && value > 80 && value < 200) {
                looks_like_skin = true;
                no_mask_score += 4;  // Very strong indicator of no mask
            }
        }
        
        // More aggressive mask-friendly saturation analysis
        if (saturation > 120) {
            no_mask_score += 4;  // Very high saturation = definitely skin
        } else if (saturation > 90) {
            no_mask_score += 2;  // High saturation = probably skin
        } else if (saturation > 60) {
            no_mask_score += 1;  // Moderate saturation = might be skin
        } else if (saturation < 50) {
            mask_score += 3;  // Low saturation = probably fabric
        } else if (saturation < 30) {
            mask_score += 4;  // Very low saturation = definitely fabric
        }
        
        // More mask-friendly texture analysis
        if (texture > 45) {
            no_mask_score += 3;  // High texture definitely suggests skin
        } else if (texture > 30) {
            no_mask_score += 1;  // Moderate texture might suggest skin
        } else if (texture < 20) {
            mask_score += 3;  // Low texture suggests uniform fabric
        } else if (texture < 12) {
            mask_score += 4;  // Very low texture = definitely uniform fabric
        }
        
        // Brightness analysis
        if (brightness > 180) {
            mask_score += 1;  // Very bright might be white mask
        } else if (brightness > 120 && brightness < 160 && looks_like_skin) {
            no_mask_score += 1;  // Typical skin brightness
        }
        
        // Enhanced mask color detection for various mask types
        bool definitely_mask_color = false;
        
        // Blue medical masks (wider range)
        if ((hue > 90 && hue < 140) && saturation < 80 && texture < 30) {
            definitely_mask_color = true;
            mask_score += 4;
        }
        // White/light colored masks (more inclusive)
        else if (saturation < 40 && value > 120) {
            if (texture < 30) {  // Relaxed texture requirement
                definitely_mask_color = true;
                mask_score += 3;
            }
        }
        // Gray masks (wider range)
        else if (saturation < 25 && value > 60 && value < 180) {
            if (texture < 25) {
                definitely_mask_color = true;
                mask_score += 3;
            }
        }
        // Black masks (more inclusive)
        else if (value < 100 && saturation < 40) {
            if (texture < 20) {
                definitely_mask_color = true;
                mask_score += 3;
            }
        }
        // Cloth masks (various colors but low saturation)
        else if (saturation < 60 && texture < 30) {
            mask_score += 2;  // Better evidence for fabric mask
        }
        // Additional mask indicators
        else if (!looks_like_skin && saturation < 70 && texture < 35) {
            mask_score += 1;  // General non-skin, low-saturation evidence
        }
        
        // Edge analysis - masks often have visible edges
        cv::Mat edges;
        cv::Canny(mouth_nose_gray, edges, 30, 100);
        int edge_pixels = cv::countNonZero(edges);
        double edge_density = (double)edge_pixels / (mouth_nose_rect.width * mouth_nose_rect.height);
        
        if (edge_density > 0.08) {  // Strong horizontal edges suggest mask boundary
            mask_score += 1;
        }
        
        // Decision based on comparative scoring - more accurate than single score
        
        // Enhanced debug logging for mask detection troubleshooting
        static int debug_counter = 0;
        if (++debug_counter % 15 == 0) {  // Log every 15 frames (twice per second)
            log_info("=== MASK DETECTION DEBUG ===");
            log_info("H=%.1f S=%.1f V=%.1f B=%.1f T=%.1f", hue, saturation, value, brightness, texture);
            log_info("SkinRatio=%.2f NonSkinRatio=%.2f", skin_ratio, non_skin_ratio);
            log_info("MaskScore=%d NoMaskScore=%d | Skin=%s | MaskColor=%s", 
                    mask_score, no_mask_score, looks_like_skin ? "YES" : "NO", 
                    definitely_mask_color ? "YES" : "NO");
            
            // Show which conditions are triggering
            log_info("Conditions: SkinHue=%s SatLow=%s TexLow=%s NonSkinHigh=%s",
                    ((hue > 5 && hue < 20) || (hue > 165 && hue < 175)) ? "YES" : "NO",
                    saturation < 30 ? "YES" : "NO",
                    texture < 15 ? "YES" : "NO",
                    non_skin_ratio > 0.3 ? "YES" : "NO");
                    
            // Show the final decision logic
            if (skin_ratio > 0.8 && no_mask_score >= 6) {
                log_info("DECISION: NO-MASK - High skin ratio (%.2f) + score (%d)", skin_ratio, no_mask_score);
            } else if (non_skin_ratio > 0.5) {
                log_info("DECISION: MASK - High non-skin ratio (%.2f)", non_skin_ratio);
            } else if (mask_score >= 5 && non_skin_ratio > 0.3) {
                log_info("DECISION: MASK - Good evidence (%d) + non-skin (%.2f)", mask_score, non_skin_ratio);
            } else if (no_mask_score >= 5 && skin_ratio > 0.75) {
                log_info("DECISION: NO-MASK - Clear evidence (%d) + skin (%.2f)", no_mask_score, skin_ratio);
            } else if (mask_score >= 4) {
                log_info("DECISION: MASK - Moderate evidence (%d)", mask_score);
            } else if (mask_score > no_mask_score + 1) {
                log_info("DECISION: MASK - Score advantage (%d vs %d)", mask_score, no_mask_score);
            } else if (no_mask_score > mask_score + 2) {
                log_info("DECISION: NO-MASK - Score advantage (%d vs %d)", no_mask_score, mask_score);
            } else {
                log_info("DECISION: %s - Tie-breaker skin ratio %.2f", skin_ratio > 0.7 ? "NO-MASK" : "MASK", skin_ratio);
            }
            log_info("============================");
        }
        
        // Balanced decision making for proper mask detection
        
        // Strong no-mask indicators first (prevent false positives)
        if (skin_ratio > 0.8 && no_mask_score >= 6) {
            return MASK_STATUS_WITHOUT_MASK;
        }
        
        // Strong mask indicators
        if (non_skin_ratio > 0.5) {
            return MASK_STATUS_WITH_MASK;
        }
        
        // Good mask evidence
        if (mask_score >= 5 && non_skin_ratio > 0.3) {
            return MASK_STATUS_WITH_MASK;
        }
        
        // Clear no-mask detection
        if (no_mask_score >= 5 && skin_ratio > 0.75) {
            return MASK_STATUS_WITHOUT_MASK;
        }
        
        // Moderate mask evidence
        if (mask_score >= 4) {
            return MASK_STATUS_WITH_MASK;
        }
        
        // Compare scores
        if (mask_score > no_mask_score + 1) {
            return MASK_STATUS_WITH_MASK;
        } else if (no_mask_score > mask_score + 2) {
            return MASK_STATUS_WITHOUT_MASK;
        } else {
            // Close call - use skin ratio as tie-breaker
            return skin_ratio > 0.7 ? MASK_STATUS_WITHOUT_MASK : MASK_STATUS_WITH_MASK;
        }
        
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception in heuristic mask classification: %s", e.what());
        return MASK_STATUS_UNKNOWN;
    }
}

// Draw detection results on frame
void draw_detections(cv::Mat& frame, const face_detection_t* faces, int count) {
    if (frame.empty() || !faces || count <= 0) {
        return;
    }
    
    for (int i = 0; i < count; i++) {
        const face_detection_t* face = &faces[i];
        
        // Choose color based on mask status
        cv::Scalar color;
        std::string label;
        
        switch (face->mask_status) {
            case MASK_STATUS_WITH_MASK:
                color = cv::Scalar(0, 255, 0); // Green
                label = "Mask";
                break;
            case MASK_STATUS_WITHOUT_MASK:
                color = cv::Scalar(0, 0, 255); // Red
                label = "No Mask";
                break;
            case MASK_STATUS_INCORRECT_MASK:
                color = cv::Scalar(0, 165, 255); // Orange
                label = "Incorrect";
                break;
            default:
                color = cv::Scalar(255, 255, 0); // Yellow
                label = "Unknown";
                break;
        }
        
        // Draw bounding box
        cv::rectangle(frame, 
                     cv::Point(face->x, face->y),
                     cv::Point(face->x + face->width, face->y + face->height),
                     color, 2);
        
        // Draw label with confidence
        char text[64];
        snprintf(text, sizeof(text), "%s (%.2f)", label.c_str(), face->mask_confidence);
        
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        
        cv::Point label_pos(face->x, face->y - 10);
        if (label_pos.y < text_size.height) {
            label_pos.y = face->y + face->height + text_size.height + 5;
        }
        
        // Draw text background
        cv::rectangle(frame,
                     cv::Point(label_pos.x, label_pos.y - text_size.height - baseline),
                     cv::Point(label_pos.x + text_size.width, label_pos.y + baseline),
                     color, -1);
        
        // Draw text
        cv::putText(frame, text, label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                   cv::Scalar(255, 255, 255), 2);
    }
    
    // Draw FPS and face count information
    char info_text[128];
    snprintf(info_text, sizeof(info_text), "Faces: %d", count);
    
    cv::putText(frame, info_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
               cv::Scalar(255, 255, 255), 2);
}

// Convert mask status to string
const char* mask_status_to_string(mask_status_t status) {
    switch (status) {
        case MASK_STATUS_WITH_MASK: return "With Mask";
        case MASK_STATUS_WITHOUT_MASK: return "Without Mask";
        case MASK_STATUS_INCORRECT_MASK: return "Incorrect Mask";
        default: return "Unknown";
    }
}

// Handle user input
int handle_key_input(app_state_t* state, int key) {
    if (!state) return FMD_ERROR_INVALID_ARGS;
    
    switch (key) {
        case 'q':
        case 27: // ESC
            state->running = false;
            return FMD_SUCCESS;
            
        case 's':
        case 'S':
            // Toggle save output
            state->config.save_output = !state->config.save_output;
            log_info("Output saving %s", state->config.save_output ? "enabled" : "disabled");
            return FMD_SUCCESS;
            
        case 'v':
        case 'V':
            // Toggle verbose mode
            state->config.verbose = !state->config.verbose;
            log_info("Verbose mode %s", state->config.verbose ? "enabled" : "disabled");
            return FMD_SUCCESS;
            
        case 'p':
        case 'P':
            // Pause/unpause
            // Implementation would depend on threading model
            log_info("Pause/unpause requested");
            return FMD_SUCCESS;
            
        case 'r':
        case 'R':
            // Reset detection parameters
            log_info("Reset detection parameters requested");
            return FMD_SUCCESS;
            
        default:
            // Unknown key
            return FMD_SUCCESS;
    }
}