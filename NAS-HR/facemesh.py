import os
import cv2
import mediapipe as mp
import struct

root_input_dir = "C:/Users/User/Documents/Monash/FYP/VIPL-HR Dataset"
root_output_dir ="C:/Users/User/Documents/Monash/FYP/health-check-with-human-face-scanner/Landmarks"

mp_face_mesh = mp.solutions.face_mesh

def process_video(input_video_path, output_folder):
    landmark_folder = os.path.join(output_folder, "face_landmarks")
    os.makedirs(landmark_folder, exist_ok=True)  # <--- where the error occurs

    cap = cv2.VideoCapture(input_video_path)
    frame_idx = 0

    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            h, w, _ = frame.shape
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            dat_filename = f"landmarks{frame_idx}.dat"
            dat_filepath = os.path.join(landmark_folder, dat_filename)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                with open(dat_filepath, "wb") as f:
                    for lm in face_landmarks.landmark:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        f.write(struct.pack("<ii", x, y))
            else:
                # If no face is detected, fill with zeros
                with open(dat_filepath, "wb") as f:
                    for _ in range(468):
                        f.write(struct.pack("<ii", 0, 0))

    cap.release()

def main():
    for dirpath, dirnames, filenames in os.walk(root_input_dir):
        # Print each directory we are visiting (for debugging)
        print("Walking in:", dirpath)

        for filename in filenames:
            if filename.lower() == "video.avi":
                input_video_path = os.path.join(dirpath, filename)

                # The relative path from root_input_dir to the current folder
                relative_path = os.path.relpath(dirpath, root_input_dir)
                # Then the final output folder
                output_folder = os.path.join(root_output_dir, relative_path)

                # Print them to see if something weird is happening
                print("  Found video:", input_video_path)
                print("  relative_path:", relative_path)
                print("  output_folder:", output_folder)

                # Try making the folder; catch any errors so we see exactly what path fails
                try:
                    os.makedirs(output_folder, exist_ok=True)
                except Exception as e:
                    print(f"  ERROR creating {output_folder}: {e}")
                    # Skip or break, but let's skip so the script continues
                    continue

                # Process the video
                process_video(input_video_path, output_folder)

    print("\nAll videos have been processed successfully!")

if __name__ == "__main__":
    main()
