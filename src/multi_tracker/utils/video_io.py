"""
Utility functions for video I/O operations in multi-animal tracking.
"""
import subprocess
import logging
from PySide2.QtCore import QThread, Signal

logger = logging.getLogger(__name__)

def create_reversed_video(input_path, output_path):
    """
    Create a reversed version of a video using FFmpeg.
    
    Args:
        input_path (str): Path to the input video file
        output_path (str): Path to save the reversed video
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # FFmpeg command to reverse video
        cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', 'reverse',
            '-c:v', 'libx264', '-crf', '18',  # Use H.264 codec with high quality
            '-an',  # Remove audio to avoid sync issues
            '-y',   # Overwrite output file if it exists
            output_path
        ]
        
        logger.info(f"Creating reversed video: {input_path} -> {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Reversed video created successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install FFmpeg to use backward tracking.")
        return False
    except Exception as e:
        logger.error(f"Error creating reversed video: {e}")
        return False

class VideoReversalWorker(QThread):
    """
    Worker thread to reverse a video using FFmpeg without blocking the GUI.
    """
    finished = Signal(bool, str, str)  # success (bool), output_path (str), error_message (str)

    def __init__(self, input_path, output_path, parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.output_path = output_path

    def run(self):
        """Execute the FFmpeg command in a separate thread."""
        try:
            cmd = [
                'ffmpeg', '-i', self.input_path,
                '-vf', 'reverse',
                '-c:v', 'libx264', '-crf', '18',
                '-an',
                '-y',
                self.output_path
            ]
            
            logger.info(f"Starting FFmpeg video reversal: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("FFmpeg reversal successful.")
            self.finished.emit(True, self.output_path, "")

        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg failed with exit code {e.returncode}:\n{e.stderr}"
            logger.error(error_msg)
            self.finished.emit(False, self.output_path, error_msg)
        except FileNotFoundError:
            error_msg = "FFmpeg not found. Please ensure FFmpeg is installed and in your system's PATH."
            logger.error(error_msg)
            self.finished.emit(False, self.output_path, error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred during video reversal: {e}"
            logger.error(error_msg)
            self.finished.emit(False, self.output_path, error_msg)