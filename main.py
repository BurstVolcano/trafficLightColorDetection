import argparse
from traffic_detector import TrafficLightDetector
from video_processor import VideoProcessor

def create_parser():
    parser = argparse.ArgumentParser(description='Traffic Light Detection System')
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--video', '-v', type=str, help='Path to video file')
    input_group.add_argument('--camera', '-c', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--frame-skip', type=int, default=1, help='Process every nth frame')
    parser.add_argument('--save', '-s', type=str, help='Save processed video to path')
    parser.add_argument('--no-display', action='store_true', help='Don\'t display video')
    return parser

def main():
    args = create_parser().parse_args()
    detector = TrafficLightDetector(model_path=args.model, confidence_threshold=args.confidence)
    processor = VideoProcessor(detector)
    source = args.video if args.video else (None if args.camera == 0 else args.camera)

    if not processor.open_source(source):
        print("Error: Could not open source")
        return 1

    info = processor.get_video_info()
    if info:
        print(f"Video info: {info['width']}x{info['height']} at {info['fps']:.1f} FPS")

    try:
        if args.save:
            if processor.save_processed_video(args.save, frame_skip=args.frame_skip):
                print("Video saved successfully!")
            else:
                print("Failed to save video")
        elif not args.no_display:
            processor.process_and_display(frame_skip=args.frame_skip)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        processor.cleanup()
    return 0

if __name__ == "__main__":
    exit(main())
