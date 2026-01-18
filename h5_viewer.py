# H5 Video Viewer
# Usage: python h5_viewer.py <input_h5_file> --camera third|gripper|all
# 'q' to quit, 'p' to pause

import h5py
import cv2
import numpy as np
import argparse

def view_h5_video(h5_file, camera_view='all'):
    try:
        with h5py.File(h5_file, 'r') as f:
            video_datasets = []
            
            # Explore structure to find videos
            def visit_item(name, obj):
                if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 3:
                    # Filter logic
                    path_lower = name.lower()
                    is_gripper = any(k in path_lower for k in ['gripper', 'eye_in_hand', 'wrist'])
                    is_third = any(k in path_lower for k in ['agentview', 'third', 'external', 'front'])
                    
                    should_include = False
                    if camera_view == 'all': should_include = True
                    elif camera_view == 'gripper' and is_gripper: should_include = True
                    elif camera_view == 'third' and is_third and not is_gripper: should_include = True
                    # Fallback for third-person if 'third' requested but heuristic missed: usually first non-gripper
                    # But keeping it simple for now as per "simple" request
                    
                    if should_include:
                         video_datasets.append((name, obj))
            
            f.visititems(visit_item)
            
            if not video_datasets:
                print("No suitable video datasets found.")
                return

            print(f"Found {len(video_datasets)} videos.")
            
            # Load frames
            frames_data = []
            max_frames = 0
            
            for name, ds in video_datasets:
                data = ds[:]
                frames_data.append((name, data))
                max_frames = max(max_frames, data.shape[0])
                
            print(f"Playing {max_frames} frames. 'q'=quit, 'p'=pause.")
            
            idx = 0
            paused = False
            
            while idx < max_frames:
                if idx > 0 and any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) <= 0 for name, _ in frames_data):
                    break

                for name, data in frames_data:
                    if idx < len(data):
                        frame = data[idx]
                        
                        if frame.dtype != np.uint8:
                            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                        
                        if len(frame.shape)==2 or frame.shape[2]==1:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        elif frame.shape[2]==3:
                            pass

                        cv2.imshow(name, frame)
                        cv2.setWindowTitle(name, f"{name} ({idx}/{max_frames})")
                
                key = cv2.waitKey(33 if not paused else 100)
                if key == ord('q'): break
                if key == ord('p'): paused = not paused
                
                if not paused: 
                    idx += 1
                    if idx % 10 == 0: print(f"Frame {idx}/{max_frames}", end='\r')
            
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_h5')
    parser.add_argument('--camera', default='all', choices=['all', 'third', 'gripper'])
    args = parser.parse_args()
    
    view_h5_video(args.input_h5, args.camera)