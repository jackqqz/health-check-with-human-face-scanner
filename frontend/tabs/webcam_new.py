import streamlit as st
import cv2
import torch
import random
import json

import pandas as pd
import time
import io
import plotly.express as px
import shutil
from utils.preprocess import extract_roi, extract_roi_age_gender
from utils.age_gender import *

from preprocess.preprocess_landmark import landmark_from_segment
from preprocess.preprocess_crop import process_segment
from preprocess.preprocess_data import gen_data
import threading
import queue
import os
OUTPUT_DIR = rOUTPUT_DIR = r"C:\Users\User\Desktop\Backup Lenovo\D Drive\Finalized_FYP\health-check-with-human-face-scanner\frontend\output"
DETECT_DIR = os.path.join(OUTPUT_DIR, "detect")

BASE = os.path.dirname(__file__)
STATS_PATH = os.path.join(BASE, "model", "error_stats.json")
try:
    stats = json.load(open(STATS_PATH, "r"))
    P95_ERROR = float(stats.get("p95_error", 1.0))
except Exception:
    P95_ERROR = 1.0

def stmap_exists(segment_id):
    stmap_path = os.path.join(
    r"C:\Users\User\Desktop\Backup Lenovo\D Drive\Finalized_FYP\health-check-with-human-face-scanner\frontend\output\data",
        segment_id,
        "POS_STMap.png"
    )
    return os.path.exists(stmap_path), stmap_path

def load_and_prepare_stmap(stmap_path, device="cuda"):
    image = cv2.imread(stmap_path)
    original_height, original_width = image.shape[:2]
    # Resize width from 140 to 300, keep height same
    new_width = 300
    new_height = original_height  # keep the height unchanged

    resized_stmap = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    resized_image = resized_stmap.astype("float32") / 255.0

    # Convert to tensor and reshape to (1, 3, 64, 300)
    tensor_input = torch.tensor(resized_image).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor_input

def predict_hr(model, stmap_tensor):
    """Run heteroscedastic model, return (hr_bpm, confidence_pct)."""
    model.eval()
    with torch.no_grad():
        mu, logvar = model(stmap_tensor)
    hr = mu.item()
    sigma = float(torch.exp(logvar).sqrt().item())

    # map [0, P95_ERROR] → [1.0 → 0.0], then to 0–100%
    conf_score = max(0.0, min(1.0, 1.0 - sigma / P95_ERROR))
    confidence = int(conf_score * 100)

    return hr, confidence
def process_video_segment(segment_folder, output_root, skin_threshold=0):
    segment_name = os.path.basename(segment_folder.rstrip("/\\"))

    # Define output directories
    landmark_dir = os.path.join(output_root, "landmark", segment_name)
    crop_dir = os.path.join(output_root, "crop", segment_name)
    data_dir = os.path.join(output_root, "data", segment_name)

    # Step 1: Extract landmarks from the video segment
    print(f"[INFO] Extracting landmarks for {segment_name}...")
    landmark_from_segment(segment_folder, landmark_dir)
    print(f"[DONE] Landmarks saved to: {landmark_dir}")

    # Step 2: Generate 5x5 crops based on landmarks
    print(f"[INFO] Generating 5x5 crops for {segment_name}...")
    process_segment(
        segment_dir=segment_folder,
        output_base_dir=crop_dir,
        landmark_dir=landmark_dir,
        skin_threshold=skin_threshold
    )
    print(f"[DONE] 5x5 crops saved to: {crop_dir}")

    # Step 3: Generate POS STMap data
    print(f"[INFO] Generating POS STMap data for {segment_name}...")
    gen_data(segment_name)
    print(f"[DONE] POS STMap data saved to: {data_dir}")

segment_task_queue = queue.Queue()

def segment_worker():
    while True:
        task = segment_task_queue.get()
        if task is None:
            break  # Optional: allows graceful shutdown
        segment_folder, output_root, skin_threshold = task
        try:
            print(f"[QUEUE] Processing segment: {segment_folder}")
            process_video_segment(segment_folder, output_root, skin_threshold)
        except Exception as e:
            print(f"[ERROR] Failed to process segment: {e}")
        finally:
            segment_task_queue.task_done()

# Start the worker thread only once
threading.Thread(target=segment_worker, daemon=True).start()

def plot_heart_rate_trend(df, hr_results):
    """
    Plots the heart rate trend over time using the provided heart rate results.

    Args:
        hr_results (list): A list of dictionaries containing heart rate data with keys:
                           - "Second": Time in seconds
                           - "Heart Rate (BPM)": Heart rate value
    """
    df["Heart Rate (BPM)"].fillna(method='ffill', inplace=True)

    # Filter out rows with missing HR values
    df_filtered = df[df["Heart Rate (BPM)"].notnull()].copy()

    st.subheader("📈 Heart Rate Trend Over Time (Live Session)")

    fig = px.line(
        df_filtered,
        x="Second",
        y="Heart Rate (BPM)",
        title="Heart Rate (BPM) by Second"
    )
    fig.update_layout(
        xaxis_title="Seconds",
        yaxis_title="Heart Rate (BPM)",
        template="plotly_dark"
    )

    # Change the line color to red
    fig.update_traces(line=dict(color="red"))

    st.plotly_chart(fig, use_container_width=True, key="webcam_hr_plot")

# Main function to run the webcam session
def webcam(model, device):
    st.subheader("📷 Real-Time Heart Rate Monitor")

    # Initialize webcam state machine
    if "webcam_state" not in st.session_state:
        st.session_state["webcam_state"] = "start"

    if "webcam_hr_results" not in st.session_state:
        st.session_state["webcam_hr_results"] = []

    if "webcam_age" not in st.session_state:
        st.session_state["webcam_age"] = "Unknown"

    if "webcam_gender" not in st.session_state:
        st.session_state["webcam_gender"] = "Unknown"

    # === Step 1: START Button ===
    if st.session_state["webcam_state"] == "start":
        st.markdown("👆 Click the **Start Webcam** button and keep your face visible. Your heart rate will be estimated within approximately 30 seconds.")
        
        if st.button("🟢 Start Webcam"):
            st.session_state["webcam_timer_start"] = time.time()
            st.session_state["webcam_state"] = "running"
            st.rerun()

        st.info("💡 This system uses your webcam to estimate heart rate by detecting tiny changes in skin color caused by blood flow. No physical contact is needed — just stay within the frame.")

    # === Step 2: STOP Button ===
    elif st.session_state["webcam_state"] == "running":
        st.markdown("🕒 The system is capturing your pulse. **Please remain still and wait at least 30 seconds** before stopping the webcam for accurate heart rate estimation.")
        
        if st.button("🛑 Stop Webcam"):
            has_valid_hr = any(
            result.get("Heart Rate (BPM)") not in [None, 0]
            for result in st.session_state.get("webcam_hr_results", [])
            )

            if not has_valid_hr:
                st.warning("⚠️ No valid heart rate was detected yet.\nPlease wait at least 30 seconds for your pulse to be measured before stopping.")
            else:
                st.session_state["webcam_state"] = "stopped"
                st.rerun()

        st.info("📡 Webcam is active. Ensure your face is clearly visible and well-lit.")


    # === Step 3: RESET Button ===
    elif st.session_state["webcam_state"] == "stopped":
        st.markdown("🔁 You have stopped the webcam. Click the **Reset Session** button below to restart the process.")

        if st.button("🔄 Reset Session"):
            # Reset all session states
            for key in ["webcam_hr_results", "webcam_age", "webcam_gender", "webcam_timer_start"]:
                st.session_state.pop(key, None)

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

            # Go back to start state
            st.session_state["webcam_state"] = "start"
            st.rerun()
        
        st.info("📁 The webcam session has ended. You can now review your heart rate trend, download the results as a CSV file, or click **Reset Session** to begin a new scan.")

    run = st.session_state["webcam_state"] == "running"

    frame_window = st.image([])
    hr_display = st.empty()
    timer_display = st.empty()

    # ── Placeholders for confidence bar (added) ──
    conf_header = st.empty()
    conf_bar = st.empty()
    conf_text = st.empty()

    stmap_counter = 0
    if run:
        cap = cv2.VideoCapture(0)

        #=========== Video Saving ===============
        segment_counter = 0
        segment_dir = os.path.join(DETECT_DIR, "detect")
        os.makedirs(segment_dir, exist_ok=True)

        video_writer = None
        segment_start_time = time.time()
        fps = 14
        #========================================

        start_time = time.time()
        minute_start = start_time
        minute_hr_values = []

        detected_age_gender = False
        last_hr_prediction = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            elapsed_time = int(time.time() - st.session_state["webcam_timer_start"])
            minutes = elapsed_time // 60
            seconds = elapsed_time % 60
            timer_display.markdown(f"⏱️ **Webcam Running:** {minutes:02d}:{seconds:02d}")

            frame = cv2.flip(frame, 1)

            # === Start or continue video segment ===
            if video_writer is None:
                segment_start_time = time.time()
                segment_name = f"segment_{segment_counter:03d}"
                segment_folder = os.path.join(DETECT_DIR, segment_name)
                detect_subdir = os.path.join(segment_folder, "detect")
                os.makedirs(detect_subdir, exist_ok=True)

                segment_path = os.path.join(segment_folder, f"{segment_name}.mp4")
                video_writer = cv2.VideoWriter(segment_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                            (frame.shape[1], frame.shape[0]))

            video_writer.write(frame)


            display_frame = frame.copy()

            frame_index = int((time.time() - segment_start_time) * fps)
            detect_dir = os.path.join(DETECT_DIR, f"segment_{segment_counter:03d}")
            rois, face_coords_list = extract_roi(frame, detect_save_dir=os.path.join(detect_dir, "detect"), frame_index=frame_index)

            stmap_name = f"segment_{stmap_counter:03d}"
            to_predict = stmap_exists(stmap_name)
            if to_predict[0]:
                stmap_path = to_predict[1]
                
                tensor_input = load_and_prepare_stmap(stmap_path, device=device)

                # Predict
                hr_prediction,confidence = predict_hr(model, tensor_input)

                                # Store last confidence so final panel can read it
                st.session_state["last_confidence"] = confidence

                # ── Show the bar right when HR arrives ──
                conf_header.subheader("🔒 Heart Rate Model Confidence Level")
                conf_bar.progress(confidence / 100)
                conf_text.markdown(f"**Confidence:** {confidence}%")

                last_hr_prediction = hr_prediction
                minute_hr_values.append(last_hr_prediction)
                stmap_counter += 1

            if rois:
                hr_display.empty()
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
                for i, (roi, face_coords) in enumerate(zip(rois, face_coords_list)):
                    x, y, w, h = face_coords
                    color = colors[i % len(colors)]

                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 1)
                    if last_hr_prediction is not None:
                        text = f"HR: {last_hr_prediction:.0f} BPM"
                    else:
                        text = "HR: calculating..."
                    cv2.putText(display_frame, text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    # processed_input = preprocess_frame(roi).to(device)
                    # with torch.no_grad():

                    # Detect age/gender once
                    if not detected_age_gender:
                        enhanced_rois, _ = extract_roi_age_gender(frame)
                        if enhanced_rois:
                            age, age_conf, gender, gender_conf = detect_age_gender(enhanced_rois[0])
                            st.session_state["webcam_age"] = age
                            st.session_state["webcam_gender"] = gender
                            st.session_state["age_conf"] = age_conf
                            st.session_state["gender_conf"] = gender_conf
                            detected_age_gender = True

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

            else:
                hr_display.warning("No faces detected. Please ensure your face is visible.\nCheck the lighting and camera angle.")
            
                conf_header.empty()
                conf_bar.empty()
                conf_text.empty()
             # Every second, save average BPM — even if no new STMap was detected
            if time.time() - minute_start >= 1:
                elapsed_time = int(time.time() - st.session_state["webcam_timer_start"])

                if minute_hr_values:
                    avg_bpm = sum(minute_hr_values) / len(minute_hr_values)
                    rounded_bpm = round(avg_bpm)
                else:
                    rounded_bpm = None  # No HR detected this second

                st.session_state["webcam_hr_results"].append({
                    "Second": elapsed_time,
                    "Heart Rate (BPM)": rounded_bpm,
                    "Age Group": st.session_state["webcam_age"],
                    "Gender": st.session_state["webcam_gender"]
                })

                minute_hr_values = []
                minute_start = time.time()

            frame_window.image(display_frame, channels="BGR")

                # ── Update confidence in the same placeholders ──
            # if last_hr_prediction is not None:   
            #     confidence = random.randint(60, 100) if rois else 0
            #     conf_header.subheader("🔒 Heart Rate Model Confidence Level")
            #     conf_bar.progress(confidence / 100)
            #     conf_text.markdown(f"**Confidence:** {confidence}%")

            if time.time() - segment_start_time >= 5:
                video_writer.release()
                print(f"[SAVED] {segment_path}")

                # Full path to current segment folder
                segment_name = f"segment_{segment_counter:03d}"
                segment_folder = os.path.join(DETECT_DIR, segment_name)
                detect_segment_folder = os.path.join(segment_folder, "detect")
                landmark_output_dir = os.path.join(OUTPUT_DIR, "landmark", segment_name)

                # Ensure the detection .pkl files exist before starting landmarking
                if os.path.exists(segment_folder):
                    try:
                        segment_task_queue.put((segment_folder, OUTPUT_DIR, 0))
                    except Exception as e:
                        print(f"[ERROR] Processing segment failed: {e}")
                else:
                    print(f"[WAITING] Detection folder not yet available: {segment_folder}")

                segment_counter += 1
                video_writer = None

        cap.release()

        if video_writer:
            video_writer.release()


    # If results exist, show graph and download 
    if st.session_state["webcam_hr_results"]:
            df = pd.DataFrame(st.session_state["webcam_hr_results"])
            plot_heart_rate_trend(df, st.session_state["webcam_hr_results"])

            # ── Confidence Level Bar ──
            last_confidence = st.session_state.get("last_confidence", 0)
            st.subheader("🔒 Heart Rate Model Confidence Level")
            st.progress(last_confidence / 100)
            st.markdown(f"**Confidence:** {last_confidence}%")

            # ── Age and Gender ──
            st.subheader("🕵️ Gender and Age Detection")
            st.markdown(
                f"**🚻 Gender:** {st.session_state['webcam_gender']} &nbsp;&nbsp;&nbsp;"
                f"**🧑 Age Group:** {st.session_state['webcam_age']}"
            )

            st.markdown(
                f"**Confidence Levels:**  \n"
                f"🟣 Gender Confidence: {st.session_state['gender_conf']:.1f}%  \n"
                f"🔵 Age Confidence: {st.session_state['age_conf']:.1f}%"
            )

            # ── Download CSV ──
            csv = io.BytesIO()
            df.to_csv(csv, index=False)
            csv.seek(0)
            st.download_button(
                label="📥 Download Webcam Results as CSV",
                data=csv,
                file_name="webcam_heart_rate_results.csv",
                mime="text/csv"
            )