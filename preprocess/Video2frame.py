import cv2
import os
import argparse
from paddleocr import PaddleOCR
import time
import re

def sanitize_filename(filename):
    """
    Sanitize a filename by replacing colon ':' characters with hyphens '-'.
    
    Args:
        filename (str): The original filename.
        
    Returns:
        str: The sanitized filename.
    """
    return filename.replace(":", "-")

def initialize_ocr():
    """
    Initialize the PaddleOCR engine with specified configurations.
    
    Returns:
        PaddleOCR: An initialized PaddleOCR instance.
    """
    # Note: Ensure CUDA is available and configured if use_gpu=True
    return PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True, show_log=False) 

def correct_filename(filename):
    """
    Correct the time format in a filename string to ensure it follows the pattern frame_XXXXXX_HH-MM-SS.jpg.
    Specifically, converts any '60' in hours, minutes, or seconds to '09'.
    Also attempts to parse less standard time formats.

    Args:
        filename (str): The input filename (e.g., 'frame_125937_1159-21.jpg' or 'frame_125937_11-59-21.jpg').

    Returns:
        str: The corrected filename with time formatted as HH-MM-SS, or the original if correction fails.
    """
    # Define the pattern to match the filename and capture groups
    pattern = re.compile(r"^(frame_\d{6}_)(\S+)\.jpg$") # More flexible for the time part

    match = pattern.match(filename)
    if not match:
        print(f"Warning: Filename format does not match expected pattern: {filename}")
        return filename

    prefix = match.group(1)  # Capture frame_XXXXXX_
    time_part_original = match.group(2)  # Capture the potentially messy time part

    # Extract all consecutive digits from the time part
    digits = re.findall(r"\d", time_part_original)
    digit_str = "".join(digits)

    # Check if we have enough digits (should be 6: HHMMSS)
    if len(digit_str) < 6:
        print(f"Warning: Insufficient digits found for time correction in: {filename}")
        return filename

    # Take the first 6 digits to form HHMMSS
    hhmmss_digits = digit_str[:6]

    # Extract hours, minutes, seconds
    hh = hhmmss_digits[0:2]
    mm = hhmmss_digits[2:4]
    ss = hhmmss_digits[4:6]

    # Correct any '60' to '09' for hours, minutes, or seconds
    hh = "09" if hh == "60" else hh
    mm = "09" if mm == "60" else mm
    ss = "09" if ss == "60" else ss

    # Construct the corrected filename
    corrected_filename = f"{prefix}{hh}-{mm}-{ss}.jpg"
    return corrected_filename

def extract_frames(video_path, output_dir, roi_coords):
    """
    Extract frames from a video, apply OCR on a specified region of interest (ROI) to get a timestamp,
    and save frames with the OCR-derived timestamp in the filename.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where the extracted frames will be saved.
        roi_coords (tuple): A tuple (x1, y1, x2, y2) defining the ROI for OCR.
                            (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner.

    Returns:
        int: The total number of frames successfully extracted and saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return 0

    frame_count = 0
    successfully_saved_count = 0
    ocr_engine = initialize_ocr()
    x1, y1, x2, y2 = roi_coords

    while True:
        try:
            # Read frame from video
            ret, frame = cap.read()
            if not ret:
                # End of video stream
                break

            # Extract the region of interest (ROI) for OCR
            # Ensure coordinates are within frame bounds (basic check)
            h, w = frame.shape[:2]
            x1_roi = max(0, min(x1, w))
            x2_roi = max(0, min(x2, w))
            y1_roi = max(0, min(y1, h))
            y2_roi = max(0, min(y2, h))

            frame[0:20, 0:45, :] = 0

            if x2_roi <= x1_roi or y2_roi <= y1_roi:
                 print(f"Warning: Invalid ROI coordinates for frame {frame_count} in {video_path}. Skipping OCR for this frame.")
                 digit_name = "invalid_roi"
            else:
                digit_img = frame[y1_roi:y2_roi, x1_roi:x2_roi, :]
                
                # Perform OCR on the ROI. det=False means we are doing recognition only.
                ocr_result = ocr_engine.ocr(digit_img, det=False)
                
                # Use default name if OCR result is empty or invalid
                # ocr_result is typically [[(text, confidence), ...], ...] or None
                if ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0 and ocr_result[0] and len(ocr_result[0]) > 0:
                    # Get the text from the first result of the first detection
                    digit_name_raw = ocr_result[0][0][0] 
                    digit_name = sanitize_filename(str(digit_name_raw))
                else:
                    digit_name = "ocr_failed"

            # Construct the base filename
            base_filename = f"frame_{frame_count:06d}_{digit_name}.jpg"

            # Apply correction logic if needed
            if len(digit_name) < 8 or "60" in digit_name:
                final_filename = correct_filename(base_filename)
            else:
                final_filename = base_filename

            # Save the full frame with the processed filename
            full_frame_filename = os.path.join(output_dir, final_filename)
            
            # Write the frame to disk
            success = cv2.imwrite(full_frame_filename, frame)
            if success:
                successfully_saved_count += 1
            else:
                 print(f"Warning: Failed to write frame {frame_count} to {full_frame_filename}")

            frame_count += 1
            
        except Exception as e:
            print(f"Error processing frame {frame_count} in {video_path}: {e}")
            # Continue processing subsequent frames
            frame_count += 1
            continue


    # Release video resources
    cap.release()
    print(f"Extraction complete for {video_path}. "
          f"Attempted: {frame_count}, Successfully saved: {successfully_saved_count}")
    return successfully_saved_count, final_filename

def main():
    """
    Main function to orchestrate the video frame extraction process.
    Parses command-line arguments, finds video files, creates output directories,
    and calls the frame extraction function for each video.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Extract video frames and name them using OCR on a timestamp region.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values
    )
    parser.add_argument('--video_root', default='../vce_data/video', 
                        help='Directory containing input video files (.mp4, .avi, .mov)')
    parser.add_argument('--frame_root', default='../vce_data/frames',
                        help='Root directory to save extracted frames in subdirectories named after videos')
    parser.add_argument('--duration_save', default='../vce_data/duration',
                        help='Root directory to save extracted frames in subdirectories named after videos')
    parser.add_argument('--x1', type=int, default=260,
                        help='X-coordinate of the top-left corner of the OCR Region of Interest (ROI)')
    parser.add_argument('--y1', type=int, default=0,
                        help='Y-coordinate of the top-left corner of the OCR Region of Interest (ROI)')
    parser.add_argument('--x2', type=int, default=320,
                        help='X-coordinate of the bottom-right corner of the OCR Region of Interest (ROI)')
    parser.add_argument('--y2', type=int, default=20,
                        help='Y-coordinate of the bottom-right corner of the OCR Region of Interest (ROI)')
    args = parser.parse_args()

    # Create the main output root directory
    os.makedirs(args.frame_root, exist_ok=True)
    os.makedirs(args.duration_save, exist_ok=True)

    # Get list of video files in the video root directory
    if not os.path.isdir(args.video_root):
        print(f"Error: Video root directory does not exist: {args.video_root}")
        return

    video_extensions = ('.mp4', '.avi', '.mov')
    video_list = [f for f in os.listdir(args.video_root)
                  if os.path.isfile(os.path.join(args.video_root, f)) and f.lower().endswith(video_extensions)]

    if not video_list:
        print(f"No video files found with extensions {video_extensions} in {args.video_root}")
        return

    print(f"Found {len(video_list)} video(s) to process.")

    # Process each video file
    for video_file in video_list:
        case_name = os.path.splitext(video_file)[0] # Get name without extension
        output_dir = os.path.join(args.frame_root, case_name)

        print(f"\n--- Processing video: {video_file} ---")
        # Create case-specific output directory
        os.makedirs(output_dir, exist_ok=True)

        # Process video and measure execution time
        start_time = time.time()
        roi_coords = (args.x1, args.y1, args.x2, args.y2)
        
        frames_saved, final_filename = extract_frames(
            os.path.join(args.video_root, video_file),
            output_dir,
            roi_coords
        )
        
        elapsed_time = time.time() - start_time
        print(f"Finished processing {video_file}. "
              f"Frames saved: {frames_saved}. Time taken: {elapsed_time:.2f} seconds.")

        duration = final_filename.split('_')[-1].replace('.jpg', '')
        duration_file = os.path.join(args.duration_save, f"{case_name}_duration.txt")
        with open(duration_file, 'w') as f:
            f.write(duration)
        print(f"Duration saved to: {duration_file}")
        
    print("\n--- All videos processed ---")

if __name__ == "__main__":
    main()