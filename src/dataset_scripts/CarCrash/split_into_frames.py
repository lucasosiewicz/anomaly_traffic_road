import cv2
from pathlib import Path
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supported video formats
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}

def get_video_paths(directory_path: str) -> List[Path]:
    """
    Get all video file paths from a directory.
    
    Args:
        directory_path (str): Path to directory containing video files
        
    Returns:
        List[Path]: List of video file paths
    """
    directory = Path(directory_path)
    
    if not directory.exists() or not directory.is_dir():
        logger.error(f"Directory not found: {directory_path}")
        return []
    
    # Use a set to avoid duplicates
    video_files_set = set()
    
    # Get all files in directory
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
            video_files_set.add(file_path)
    
    video_files = sorted(list(video_files_set))
    logger.info(f"Found {len(video_files)} video files in {directory_path}")
    
    return video_files

def extract_frames_from_video(video_path: Path, output_dir: str = "frames", frame_skip: int = 1, quality: int = 95) -> int:
    """
    Extract frames from a single video file and save as JPG images.
    
    Args:
        video_path (Path): Path to the video file
        output_dir (str): Base directory to save extracted frames
        frame_skip (int): Extract every nth frame (1 = all frames, 2 = every 2nd frame, etc.)
        quality (int): JPG quality (0-100, higher is better quality)
        
    Returns:
        int: Number of frames extracted
    """
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return 0
    
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        logger.warning(f"Unsupported video format: {video_path.suffix}")
        return 0
    
    # Create output directory structure
    output_base = Path(output_dir)
    video_output_dir = output_base / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        logger.error(f"Error opening video: {video_path}")
        return 0
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    logger.info(f"Processing video: {video_path.name}")
    logger.info(f"Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Extract frame based on skip interval
        if frame_count % frame_skip == 0:
            # Generate frame filename
            frame_filename = f"frame_{extracted_count:06d}.jpg"
            frame_path = video_output_dir / frame_filename
            
            # Save frame as JPG
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            extracted_count += 1
            
            if extracted_count % 100 == 0:
                logger.info(f"Extracted {extracted_count} frames...")
        
        frame_count += 1
    
    cap.release()
    logger.info(f"Successfully extracted {extracted_count} frames from {video_path.name}")
    
    return extracted_count

def convert_videos_to_frames(videos_directory: str, output_dir: str = "frames", frame_skip: int = 1, quality: int = 95) -> int:
    """
    Main function to convert all videos in a directory to frames.
    
    Args:
        videos_directory (str): Path to directory containing video files
        output_dir (str): Directory to save extracted frames (default: "frames")
        frame_skip (int): Extract every nth frame (1 = all frames, 2 = every 2nd frame, etc.)
        quality (int): JPG quality (0-100, higher is better quality)
        
    Returns:
        int: Total number of frames extracted from all videos
    """
    logger.info(f"Starting video to frames conversion from directory: {videos_directory}")
    
    # Get all video paths
    video_paths = get_video_paths(videos_directory)
    
    if not video_paths:
        logger.warning("No video files found to process")
        return 0
    
    total_extracted = 0
    
    # Process each video one by one
    for i, video_path in enumerate(video_paths, 1):
        logger.info(f"Processing video {i}/{len(video_paths)}: {video_path.name}")
        extracted = extract_frames_from_video(video_path, output_dir, frame_skip, quality)
        total_extracted += extracted
        logger.info(f"Completed {i}/{len(video_paths)} videos")
    
    logger.info(f"Conversion complete! Total frames extracted: {total_extracted}")
    return total_extracted

def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: Video information
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return {}
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info

# Example usage
if __name__ == "__main__":
    # Example: Convert all videos in a directory to frames
    videos_dir = r"D:\MAGISTERKA\anomaly_traffic_road\datasets\CarCrash\anomaly"  # Replace with your videos directory
    output_frames_dir = "frames/train"  # Output directory for frames
    
    # Convert all videos to frames (extract every frame)
    total_frames = convert_videos_to_frames(
        videos_directory=videos_dir,
        output_dir=output_frames_dir,
        frame_skip=1,  # Extract every frame
        quality=95     # High quality JPG
    )
    
    print(f"Extraction complete. Total frames: {total_frames}")
