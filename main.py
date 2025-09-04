import argparse
from traffic_detector import TrafficLightDetector
from video_processor import VideoProcessor


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Traffic Light Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Use webcam
  python main.py -v video.mp4             # Process video file
  python main.py -v video.mp4 -s output.mp4  # Save processed video
  python main.py -c 1 --confidence 0.5    # Use camera 1 with 50% confidence
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--video', '-v', type=str,
                             help='Path to video file')
    input_group.add_argument('--camera', '-c', type=int, default=0,
                             help='Camera index (default: 0)')

    # Processing options
    parser.add_argument('--confidence', type=float, default=0.3,
                        help='Confidence threshold (default: 0.3)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLO model path (default: yolov8n.pt)')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Process every nth frame (default: 1)')

    # Output options
    parser.add_argument('--save', '-s', type=str,
                        help='Save processed video to path')
    parser.add_argument('--no-display', action='store_true',
                        help='Don\'t display video (useful with --save)')

    return parser


def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()

    # Initialize detector
    print(f"Loading YOLO model: {args.model}")
    detector = TrafficLightDetector(
        model_path=args.model,
        confidence_threshold=args.confidence
    )

    # Initialize video processor
    processor = VideoProcessor(detector)

    # Determine input source
    if args.video:
        source = args.video
        print(f"Using video file: {args.video}")
    else:
        source = None if args.camera == 0 else args.camera
        print(f"Using camera (index: {args.camera})")

    # Open video source
    if not processor.open_source(source):
        print(f"Error: Could not open {'video file' if args.video else 'camera'}")
        if not args.video:
            print("Make sure your camera is connected and not being used by another application")
        return 1

    # Display video info
    info = processor.get_video_info()
    if info:
        print(f"Video info: {info['width']}x{info['height']} at {info['fps']:.1f} FPS")
        if 'total_frames' in info and info['total_frames'] > 0:
            print(f"Total frames: {info['total_frames']}")

    try:
        # Save processed video if requested
        if args.save:
            print(f"Processing and saving to: {args.save}")
            success = processor.save_processed_video(
                args.save,
                frame_skip=args.frame_skip
            )
            if success:
                print("Video saved successfully!")
            else:
                print("Failed to save video")
                return 1

        # Display video if not disabled
        if not args.no_display and not args.save:
            processor.process_and_display(frame_skip=args.frame_skip)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as e:
        print(f"Error during processing: {e}")
        return 1

    finally:
        processor.cleanup()

    return 0


if __name__ == "__main__":
    exit(main())