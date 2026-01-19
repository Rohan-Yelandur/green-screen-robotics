import h5py
import cv2
import numpy as np
import argparse
import os

def extract_video(h5_file, output_file=None):
    if not os.path.exists(h5_file):
        print(f"Error: File {h5_file} not found.")
        return

    try:
        with h5py.File(h5_file, 'r') as f:
            target_ds = None
            target_name = ""
            
            # Find the best candidate dataset
            def visit_item(name, obj):
                nonlocal target_ds, target_name
                if target_ds is not None: return # Already found one
                
                if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 3:
                    path_lower = name.lower()
                    # Priority Keywords for Third Person
                    is_third = any(k in path_lower for k in ['agentview', 'third', 'external', 'front', 'primary'])
                    is_gripper = any(k in path_lower for k in ['gripper', 'eye_in_hand', 'wrist'])
                    
                    if is_third and not is_gripper:
                        target_ds = obj
                        target_name = name
            
            f.visititems(visit_item)
            
            if target_ds is None:
                print("No suitable third-person video dataset found.")
                return

            print(f"Found video dataset: {target_name}")
            data = target_ds[:]
            
            # Prepare Output
            if output_file is None:
                base_name = os.path.splitext(os.path.basename(h5_file))[0]
                output_file = f"{base_name}.mp4"
            
            num_frames, height, width = data.shape[:3]
            fps = 30 # Default guess
            
            # Setup Video Writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            
            print(f"Writing {num_frames} frames to {output_file}...")
            
            for i in range(num_frames):
                frame = data[i]
                
                # Normalize if float
                if frame.dtype != np.uint8:
                     frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                
                # Color conversion
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 3:
                     pass # H5 is already BGR (verified via h5_viewer)
                
                out.write(frame)
                
            out.release()
            print("Done.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract third-person video from H5 file to MP4.")
    parser.add_argument("h5_file", help="Path to input H5 file")
    parser.add_argument("-o", "--output", help="Path to output MP4 file (optional)")
    args = parser.parse_args()
    
    extract_video(args.h5_file, args.output)
