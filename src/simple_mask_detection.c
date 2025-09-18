#include "face_mask_detector.h"
#include "image_processing.h"

// Classify whether a face is wearing a mask using simple reliable method
mask_status_t classify_mask_simple_reliable(const cv::Mat& frame, const face_detection_t* face) {
    if (frame.empty() || !face) {
        return MASK_STATUS_UNKNOWN;
    }
    
    try {
        // Get face region from frame
        cv::Rect face_region(face->x, face->y, face->width, face->height);
        face_region &= cv::Rect(0, 0, frame.cols, frame.rows);
        
        // Skip tiny faces
        if (face_region.width < 20 || face_region.height < 20) {
            return MASK_STATUS_UNKNOWN;
        }
        
        cv::Mat face_img = frame(face_region);
        
        // Focus on lower face where masks are visible
        int lower_y = face_img.rows * 0.65;
        int lower_height = face_img.rows * 0.25;
        cv::Rect lower_face(face_img.cols * 0.2, lower_y, face_img.cols * 0.6, lower_height);
        lower_face &= cv::Rect(0, 0, face_img.cols, face_img.rows);
        
        if (lower_face.width < 10 || lower_face.height < 10) {
            return MASK_STATUS_UNKNOWN;
        }
        
        cv::Mat mouth_area = face_img(lower_face);
        
        // Convert to different color spaces for analysis
        cv::Mat hsv_img, gray_img;
        cv::cvtColor(mouth_area, hsv_img, cv::COLOR_BGR2HSV);
        cv::cvtColor(mouth_area, gray_img, cv::COLOR_BGR2GRAY);
        
        // Score different indicators
        int mask_indicators = 0;
        int skin_indicators = 0;
        
        // 1. Check color saturation - masks usually have lower saturation
        cv::Scalar color_mean = cv::mean(hsv_img);
        double saturation = color_mean[1];
        double hue = color_mean[0];
        double value = color_mean[2];
        
        if (saturation < 70) {
            mask_indicators += 4;  // Low saturation suggests mask material
        } else if (saturation > 90) {
            skin_indicators += 2;  // High saturation more like skin
        }
        
        // 2. Skin color detection - more restrictive to avoid false no-mask detections
        bool is_skin_hue = ((hue >= 5 && hue <= 25) || (hue >= 165 && hue <= 175));
        if (is_skin_hue && saturation > 60 && value > 80 && value < 180) {
            skin_indicators += 3;  // Definitely skin-like (stricter requirements)
        }
        
        // 3. Texture uniformity
        cv::Scalar mean_gray, std_gray;
        cv::meanStdDev(gray_img, mean_gray, std_gray);
        double texture_std = std_gray[0];
        
        if (texture_std < 25) {
            mask_indicators += 3;  // Very uniform = mask (more generous)
        } else if (texture_std > 35) {
            skin_indicators += 1;  // Varied texture = skin (higher threshold)
        }
        
        // 4. Overall brightness uniformity
        double brightness = mean_gray[0];
        if (brightness > 150 || brightness < 100) {
            // Very bright or dark suggests mask
            mask_indicators += 1;
        }
        
        // 5. Edge analysis for mask boundaries
        cv::Mat edges;
        cv::Canny(gray_img, edges, 50, 150);
        int edge_pixels = cv::countNonZero(edges);
        double edge_ratio = (double)edge_pixels / (lower_face.width * lower_face.height);
        
        if (edge_ratio > 0.1 && edge_ratio < 0.3) {
            mask_indicators += 1;  // Moderate edges suggest mask boundary
        }
        
        // Debug output
        static int debug_counter = 0;
        if (++debug_counter % 15 == 0) {
            log_info("=== SIMPLE RELIABLE DETECTION ===");
            log_info("H=%.1f S=%.1f V=%.1f B=%.1f T=%.1f", hue, saturation, value, brightness, texture_std);
            log_info("MaskIndicators=%d SkinIndicators=%d", mask_indicators, skin_indicators);
            log_info("SkinHue=%s EdgeRatio=%.3f", is_skin_hue ? "YES" : "NO", edge_ratio);
            
            if (mask_indicators >= 4) {
                log_info("DECISION: MASK (strong indicators >= 4)");
            } else if (skin_indicators >= 4) {
                log_info("DECISION: NO-MASK (strong skin indicators >= 4)");
            } else if (mask_indicators > skin_indicators) {
                log_info("DECISION: MASK (advantage %d > %d)", mask_indicators, skin_indicators);
            } else {
                log_info("DECISION: NO-MASK (advantage %d >= %d)", skin_indicators, mask_indicators);
            }
            log_info("================================");
        }
        
        // Mask-friendly decision logic
        if (mask_indicators >= 3) {
            return MASK_STATUS_WITH_MASK;
        } else if (skin_indicators >= 5) {
            return MASK_STATUS_WITHOUT_MASK;
        } else if (mask_indicators >= skin_indicators) {
            return MASK_STATUS_WITH_MASK;
        } else {
            return MASK_STATUS_WITHOUT_MASK;
        }
        
    } catch (const cv::Exception& e) {
        log_error("OpenCV exception in simple mask classification: %s", e.what());
        return MASK_STATUS_UNKNOWN;
    }
}