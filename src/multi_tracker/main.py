#!/usr/bin/env python3
"""
Main entry point for the Multi-Animal Tracker application.

This module provides the command-line interface and GUI launcher for the
multi-animal tracking system.
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Fix OpenMP conflict on macOS (PyTorch + OpenCV + NumPy can load multiple OpenMP libraries)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import Qt before other modules to avoid conflicts
try:
    from PySide2.QtWidgets import QApplication
    from PySide2.QtCore import Qt
except ImportError as e:
    print("Error: PySide2 not found. Please install it with:")
    print("conda install -c conda-forge pyside2")
    print("or")
    print("pip install PySide2")
    sys.exit(1)


# Set up logging
def setup_logging(log_level=logging.INFO, enable_file_logging=False, log_dir=None):
    """Set up logging configuration for the multi-tracker application.

    Note: File logging is now handled per-session in main_window.py.
    This only sets up console logging.
    """

    # Only set up console logging - session logs are created in main_window.py
    handlers = [logging.StreamHandler(sys.stdout)]

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )

    # Log startup info
    logger = logging.getLogger(__name__)
    logger.info("Multi-Animal Tracker starting up...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Animal Tracker - Real-time animal tracking with circular ROI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  multianimaltracker                    # Launch GUI
  multianimaltracker --log-level DEBUG # Launch with debug logging
  multianimaltracker --no-file-log     # Disable file logging
        """,
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    parser.add_argument(
        "--no-file-log", action="store_true", help="Disable file logging (console only)"
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory for log files (default: current directory)",
    )

    parser.add_argument(
        "--version", action="version", version="Multi-Animal Tracker 1.0.0"
    )

    return parser.parse_args()


def check_dependencies():
    """Check that all required dependencies are available."""
    required_modules = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy"),
        ("skimage", "scikit-image"),
    ]

    missing_modules = []
    for module_name, package_name in required_modules:
        try:
            __import__(module_name)
        except ImportError:
            missing_modules.append(package_name)

    if missing_modules:
        print("Error: Missing required dependencies:")
        for package in missing_modules:
            print(f"  - {package}")
        print("\nPlease install missing packages with:")
        print(f"conda install -c conda-forge {' '.join(missing_modules)}")
        print("or")
        print(f"pip install {' '.join(missing_modules)}")
        return False

    return True


def main():
    """
    Application entry point.

    Parses command line arguments, sets up logging, checks dependencies,
    creates Qt application, initializes main window, and starts event loop.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(
        log_level=log_level,
        enable_file_logging=not args.no_file_log,
        log_dir=args.log_dir,
    )

    logger = logging.getLogger(__name__)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    try:
        # Import GUI components (after dependency check)
        from .gui.main_window import MainWindow

        # Create Qt application
        app = QApplication(sys.argv)
        app.setApplicationName("Multi-Animal Tracker")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("MBL")

        # Set application icon if available
        try:
            from PySide2.QtGui import QIcon

            icon_path = Path(__file__).parent / "resources" / "icon.png"
            if icon_path.exists():
                app.setWindowIcon(QIcon(str(icon_path)))
        except Exception:
            pass  # Icon not critical

        # Create and show main window
        logger.info("Initializing main window...")
        main_window = MainWindow()
        main_window.show()

        logger.info("Multi-Animal Tracker GUI launched successfully")

        # Start Qt event loop
        exit_code = app.exec_()
        logger.info(f"Application exited with code {exit_code}")
        sys.exit(exit_code)

    except ImportError as e:
        logger.error(f"Failed to import GUI components: {e}")
        print(f"Error: Failed to load GUI components: {e}")
        print(
            "Make sure all dependencies are installed and the package is properly installed."
        )
        sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error during startup: {e}", exc_info=True)
        print(f"Error: Unexpected error during startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
