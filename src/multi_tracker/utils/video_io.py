"""
Utility functions for video I/O operations in multi-animal tracking.
"""
import subprocess
import logging
import os

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
