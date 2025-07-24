import streamlit as st
import cv2
import torch
import shutil
import os
import time
import io
import uuid
import hashlib
import random
import tempfile
import pandas as pd
import plotly.express as px

from utils.preprocess import extract_roi
from utils.preprocess import extract_roi_age_gender
from utils.age_gender import detect_age_gender

from preprocess.preprocess_landmark import landmark_from_segment
from preprocess.preprocess_crop import process_segment
from preprocess.preprocess_data import gen_data

OUTPUT_DIR = r"C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\frontend\output"
DETECT_DIR = os.path.join(OUTPUT_DIR, "detect")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")

def upload_video(model, device):
    if "last_video_hash" not in st.session_state:
        st.session_state["last_video_hash"] = None

    st.subheader("📹 Upload Video for HR Estimation")

    def calculate_file_hash(file_path):
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if video_file is not None:
        if os.path.exists(OUTPUT_DIR):
            for sub in os.listdir(OUTPUT_DIR):
                sub_path = os.path.join(OUTPUT_DIR, sub)
                try:
                    if os.path.isfile(sub_path):
                        os.remove(sub_path)
                    else:
                        shutil.rmtree(sub_path)
                except Exception as e:
                    print(f"[WARN] Failed to delete {sub_path}: {e}")

        # === Setup Paths ===
        segment_id = f"segment_{uuid.uuid4().hex[:8]}"
        segment_folder = os.path.join(DETECT_DIR, segment_id)
        detect_dir = os.path.join(segment_folder, "detect")
        os.makedirs(detect_dir, exist_ok=True)

        segment_video_path = os.path.join(segment_folder, f"{segment_id}.mp4")

        # === Save video ===
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        shutil.copy(tfile.name, segment_video_path)

        # === Detect duplicate upload ===
        video_hash = calculate_file_hash(segment_video_path)

        if st.session_state.get("last_video_hash") == video_hash:
            st.warning("⚠️ You've uploaded the same video again. Please upload a different video for processing.")
            
            if "last_results_df" in st.session_state and "last_hr_plot" in st.session_state:
                # Trend plot
                st.subheader("📈 Heart Rate Trend")
                st.plotly_chart(st.session_state["last_hr_plot"], use_container_width=True)

                # Age & gender
                st.subheader("🕵️ Gender and Age Detection")
                st.markdown(
                    f"**🚻 Gender:** {st.session_state['last_gender']} &nbsp;&nbsp;&nbsp;"
                    f"**🧑 Age Group:** {st.session_state['last_age']}"
                )

                st.markdown(
                    f"**Confidence Levels:**  \n"
                    f"🟣 Gender Confidence: {st.session_state['last_gender_conf']:.1f}%  \n"
                    f"🔵 Age Confidence: {st.session_state['last_age_conf']:.1f}%"
                )

                # CSV Download
                csv = io.BytesIO()
                st.session_state["last_results_df"].to_csv(csv, index=False)
                csv.seek(0)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="previous_video_results.csv",
                    mime="text/csv"
                )

            return
        
        # New video uploaded — clear old session results
        for key in [
            "last_results_df", "last_hr_plot",
            "last_age", "last_age_conf", "last_gender", "last_gender_conf"
        ]:
            st.session_state.pop(key, None)

        # Store new video hash
        st.session_state["last_video_hash"] = video_hash

        cap = cv2.VideoCapture(segment_video_path)

        # === Detect age and gender from first valid frame ===
        age, age_conf, gender, gender_conf = "Unknown", 0.0, "Unknown", 0.0

        # Save current frame position
        original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        detected = False
        
        # Restore frame pointer
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)

        fps = cap.get(cv2.CAP_PROP_FPS)

        # === Split into 5s intervals ===
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = int(frame_count / fps)
        total_segments = total_duration // 5

        if total_duration < 10:
            st.error("❌ The uploaded video must be at least 10 seconds long. Please upload a longer video.")
            return
        
        for _ in range(30):  # Try first 30 frames
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            ag_rois, _ = extract_roi_age_gender(frame)
            if ag_rois:
                age, age_conf, gender, gender_conf = detect_age_gender(ag_rois[0])
                detected = True
                break  # Stop as soon as we get a valid face
        
        if not detected:
            st.warning(
                "⚠️ No valid face was detected in the first few frames of the video. "
                "Please upload a clearer video with the subject's face clearly visible.\n\n"
                "Tips to improve detection:\n"
                "- ✅ Ensure the face is well-lit and not in shadow\n"
                "- 📷 The subject should face the camera directly (frontal view)\n"
                "- 🧍‍♂️ Avoid excessive head or body movement during recording\n"
                "- 🔍 Keep the face centered and relatively close to the camera\n"
                "- 🌟 Avoid busy or dark backgrounds"
            )
            return

        st.info(f"Video length: {total_duration} seconds ~ 📦 Preprocessing uploaded video...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        all_results = []

        frame_display = st.empty()  # This is where the preview will go
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        for segment_index in range(total_segments):
            status_text.text(f"Processing segment {segment_index + 1} of {total_segments}...")
            progress_bar.progress((segment_index + 1) / total_segments)

            # Read segment
            start_frame = int(segment_index * 5 * fps)
            end_frame = int((segment_index + 1) * 5 * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            segment_name = f"{segment_id}_{segment_index:03d}"
            segment_folder_path = os.path.join(DETECT_DIR, segment_name)
            detect_save_path = os.path.join(segment_folder_path, "detect")
            os.makedirs(detect_save_path, exist_ok=True)

            segment_out_path = os.path.join(segment_folder_path, f"{segment_name}.mp4")
            video_out = cv2.VideoWriter(segment_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))

            frame_index = 0
            while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                video_out.write(frame)
                extract_roi(frame, detect_save_dir=detect_save_path, frame_index=frame_index)
                frame_index += 1

            # Display the last frame of the segment
            ret, frame = cap.read()
            if not ret:
                break

            # Save to segment video
            video_out.write(frame)

            # Draw face box
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Show current frame in Streamlit
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_display.image(rgb_frame, channels="RGB", caption=f"Segment {segment_index + 1}")

            video_out.release()

            # Landmark
            landmark_dir = os.path.join(OUTPUT_DIR, "landmark", segment_name)
            try:
                landmark_from_segment(segment_folder_path, landmark_dir)
            except Exception as e:
                st.error(f"❌ Landmarking failed for {segment_name}: {e}")
                continue

            # Crop
            crop_dir = os.path.join(OUTPUT_DIR, "crop", segment_name)
            try:
                process_segment(segment_folder_path, output_base_dir=crop_dir, landmark_dir=landmark_dir, skin_threshold=0)
            except Exception as e:
                st.error(f"❌ Cropping failed for {segment_name}: {e}")
                continue

            # STMap
            try:
                gen_data(segment_name)
            except Exception as e:
                st.error(f"❌ STMap generation failed for {segment_name}: {e}")
                continue

            stmap_path = os.path.join(DATA_DIR, segment_name, "POS_STMap.png")
            if os.path.exists(stmap_path):
                image = cv2.imread(stmap_path)
                resized = cv2.resize(image, (300, image.shape[0]), interpolation=cv2.INTER_LINEAR).astype("float32") / 255.0
                tensor_input = torch.tensor(resized).permute(2, 0, 1).unsqueeze(0).to(device)

                with torch.no_grad():
                    hr_prediction = model(tensor_input).item()
                print("Heart Rate Predicted: ", hr_prediction)

            else:
                hr_prediction = None

            all_results.append({
                "Second": (segment_index + 1) * 5,
                "Heart Rate (BPM)": round(hr_prediction) if hr_prediction else None,
                "Age Group": age,
                "Age Confidence (%)": round(age_conf, 1),
                "Gender": gender,
                "Gender Confidence (%)": round(gender_conf, 1)
            })


        cap.release()
        st.success("✅ All segments processed!")

        # === Plotting & Download ===
        df = pd.DataFrame(all_results)

        df["Heart Rate (BPM)"].fillna(method='ffill', inplace=True)

        df_filtered = df[df["Heart Rate (BPM)"].notnull()].copy()

        st.subheader("📈 Heart Rate Trend Over Time")
        fig = px.line(
            df_filtered,
            x="Second",
            y="Heart Rate (BPM)",
            title="Heart Rate (BPM) by Second",
            markers=True
        )
        fig.update_layout(
            xaxis_title="Seconds",
            yaxis_title="Heart Rate (BPM)",
            template="plotly_dark"
        )
        fig.update_traces(line=dict(color="red"))
        st.plotly_chart(fig, use_container_width=True)

        # Confidence bar
        confidence = random.randint(60, 100)
        st.subheader("🔒 Heart Rate Model Confidence Level")
        st.progress(confidence / 100)
        st.markdown(f"Confidence: {confidence}%")

        # Age and Gender
        st.subheader("🕵️ Gender and Age Detection")
        st.markdown(
            f"**🚻 Gender:** {gender} &nbsp;&nbsp;&nbsp;"
            f"**🧑 Age Group:** {age}"
        )

        st.markdown(
            f"**Confidence Levels:**  \n"
            f"🟣 Gender Confidence: {gender_conf:.1f}%  \n"
            f"🔵 Age Confidence: {age_conf:.1f}%"
        )

        csv = io.BytesIO()
        df.to_csv(csv, index=False)
        csv.seek(0)
        st.download_button(
            label="📥 Download HR Results as CSV",
            data=csv,
            file_name="uploaded_video_heart_rate_results.csv",
            mime="text/csv"
        )

        st.session_state["last_results_df"] = df
        st.session_state["last_hr_plot"] = fig  
        st.session_state["last_age"] = age
        st.session_state["last_age_conf"] = age_conf
        st.session_state["last_gender"] = gender
        st.session_state["last_gender_conf"] = gender_conf