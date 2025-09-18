#!/bin/bash

# Face Mask Detection App

echo "Face Mask Detection"
echo "==================="
echo ""
echo "Starting camera and detection..."
echo ""
echo "Controls:"
echo "  Press 'q' to quit"
echo "  Press 'v' for debug info"
echo ""
echo "Make sure you have good lighting and are facing the camera."
echo ""

# Run the main application
./bin/face_mask_detector -c config/face_mask_detector.conf -v

echo ""
echo "Detection stopped."
