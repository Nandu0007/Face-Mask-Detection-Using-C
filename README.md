# Face Mask Detection

A practical computer vision tool that detects whether someone is wearing a face mask in real-time using your camera.

## What it does

This app opens your camera and shows a live video feed with colored rectangles around detected faces:
- **Green box**: Person is wearing a mask
- **Red box**: Person is not wearing a mask  
- **Blue box**: Can't determine (face too small, poor lighting, etc.)

The detection works by analyzing facial features, colors, and textures in the lower part of the face where masks typically sit.

## Getting started

**Prerequisites**: You'll need OpenCV installed. On macOS with Homebrew:
```bash
brew install opencv
```

**Build and run**:
```bash
make
./bin/face_mask_detector
```

Press 'q' to quit the application.

## Different detection modes

The app includes several test scripts for different scenarios:

- `./test_simple.sh` - Basic detection, good for most cases
- `./test_anti_flicker.sh` - More stable detection that reduces flickering
- `./test_ultra_stable.sh` - Maximum stability, slower to change but very consistent

Try different modes to see what works best for your setup and lighting conditions.

## How it works

The detection combines several computer vision techniques:
- Face detection using Haar cascade classifiers
- Color analysis in HSV space to distinguish mask materials from skin
- Texture analysis to identify uniform mask surfaces vs varied skin texture
- Brightness and edge detection for additional clues
- Temporal smoothing to avoid rapid status changes

The system is designed to work well even with glasses, different lighting, and various mask colors and styles.

## Project structure

```
face-mask-detection-c/
├── bin/face_mask_detector          # Main executable
├── src/                            # Source code
├── include/                        # Header files
├── models/                         # Face detection models
├── config/                         # Configuration file
└── Makefile                        # Build instructions
```

## Troubleshooting

**Camera not working**: Make sure you've granted camera permissions to Terminal.

**No faces detected**: Try better lighting and face the camera directly. Works best at arm's length distance.

**Inconsistent detection**: Use one of the stability modes (anti-flicker or ultra-stable) for more consistent results.

**Build errors**: Make sure OpenCV is properly installed with `brew install opencv`.
