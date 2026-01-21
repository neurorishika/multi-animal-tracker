# Changelog

All notable changes to the Multi-Animal Tracker project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **YOLO GUI Integration**: Full graphical user interface support for YOLO detection
  - Detection method dropdown in main window (Background Subtraction / YOLO OBB)
  - YOLO model selection dropdown with YOLOv8 and YOLOv11 models
  - Custom model file browser
  - Confidence threshold slider
  - IOU threshold slider  
  - Target classes text input for filtering specific COCO classes
  - Dark mode styling for all new controls
  - Automatic show/hide of YOLO parameters based on detection method
- **YOLO OBB Detection Support**: Added support for YOLO Oriented Bounding Box detection as an alternative to background subtraction
  - New `YOLOOBBDetector` class in `detection.py`
  - Factory function `create_detector()` to select detection method
  - Configuration options: `detection_method`, `yolo_model_path`, `yolo_confidence_threshold`, `yolo_iou_threshold`, `yolo_target_classes`
  - Support for both pretrained and custom YOLO models
  - Automatic model downloading for pretrained models
  - Support for YOLOv11 models (latest version)
- **Documentation**: 
  - Added comprehensive YOLO detection guide (`docs/yolo_detection_guide.md`)
  - Added GUI-specific guide (`docs/yolo_gui_guide.md`)
  - Quick reference card (`YOLO_QUICK_REFERENCE.md`)
- **Examples**: Added example script for YOLO configuration (`examples/yolo_detection_example.py`)
- **Dependencies**: Added ultralytics to pip dependencies in `environment.yml`

### Changed
- Updated `main_window.py` with YOLO detection controls and UI integration
- Enhanced `QComboBox` styling for dark mode theme
- Updated `tracking_worker.py` to support both detection methods
- Modified frame processing loop to handle YOLO and background subtraction separately
- Background model initialization now conditional based on detection method
- Updated `load_config` and `save_config` to handle YOLO parameters
- Updated README.md with YOLO GUI usage instructions
- Enhanced `get_parameters_dict` to include YOLO configuration

### Technical Details
- YOLO detections provide oriented bounding boxes (OBB) compatible with existing tracking pipeline
- Detection interface remains consistent between methods for seamless integration
- YOLO detection bypasses background subtraction pipeline when selected
- Maintains backward compatibility with existing background subtraction workflows

---

## Development Guidelines

### 1. Define the Purpose
- **What does the tool do?** Clearly outline the functionality and objectives of your tool.
- **Who is the target audience?** Identify who will be using the tool and tailor the organization accordingly.

### 2. Structure the Codebase
- **Directory Structure:**
  - **/src**: Main source code files.
  - **/tests**: Unit tests and integration tests.
  - **/docs**: Documentation files.
  - **/examples**: Example scripts or usage scenarios.
  - **/assets**: Any additional resources (images, data files, etc.).
  - **/bin**: Executable scripts or command-line interfaces.

### 3. Modularize the Code
- **Functions/Classes**: Break down the script into functions or classes based on functionality.
- **Modules**: Group related functions/classes into modules (Python files) to enhance readability and reusability.

### 4. Documentation
- **README.md**: Provide an overview of the tool, installation instructions, usage examples, and contribution guidelines.
- **Docstrings**: Use docstrings in your functions and classes to explain their purpose, parameters, and return values.
- **API Documentation**: If applicable, generate API documentation using tools like Sphinx or JSDoc.

### 5. Configuration
- **Configuration Files**: Use configuration files (e.g., JSON, YAML, or INI) for settings that users might want to customize.
- **Environment Variables**: Consider using environment variables for sensitive information or settings that may change between environments.

### 6. Testing
- **Unit Tests**: Write unit tests for individual functions/classes to ensure they work as expected.
- **Integration Tests**: Test how different parts of the tool work together.
- **Continuous Integration**: Set up CI/CD pipelines to automate testing and deployment.

### 7. Version Control
- **Git**: Use Git for version control. Create a `.gitignore` file to exclude unnecessary files.
- **Branching Strategy**: Define a branching strategy (e.g., Git Flow) for managing features, fixes, and releases.

### 8. Distribution
- **Package Management**: If applicable, create a package for distribution (e.g., PyPI for Python, npm for JavaScript).
- **Installation Instructions**: Provide clear instructions on how to install and use the tool.

### 9. Community and Support
- **Contribution Guidelines**: Outline how others can contribute to the project.
- **Issue Tracker**: Use an issue tracker (like GitHub Issues) for bug reports and feature requests.
- **Discussion Forum**: Consider setting up a forum or chat (like Discord or Slack) for user support and community engagement.

### Example Directory Structure
```
my_tool/
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── module1.py
│   └── module2.py
│
├── tests/
│   ├── test_module1.py
│   └── test_module2.py
│
├── docs/
│   ├── README.md
│   └── usage_guide.md
│
├── examples/
│   └── example_usage.py
│
├── assets/
│   └── sample_data.csv
│
├── bin/
│   └── run_tool.sh
│
├── .gitignore
├── requirements.txt
└── setup.py
```

### Conclusion
By following this structure, you can create a well-organized tool that is easy to use, maintain, and share with others. If you have specific details about your script or tool, feel free to share, and I can provide more tailored advice!