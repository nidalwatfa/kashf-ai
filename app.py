import gradio as grtry:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if not TORCH_AVAILABLE:
    print("⚠️ مكتبة Torch غير متاحة على هذه البيئة. سيتم تشغيل الوضع التجريبي فقط.")
import torch
import cv2
import numpy as np
import librosa
from transformers import pipeline, AutoModelForAudioClassification, AutoFeatureExtractor
import tempfile
import os
from moviepy.editor import VideoFileClip
import warnings
warnings.filterwarnings("ignore")

# ------------------- نموذج كشف التزييف في الصوت (عربي) -------------------
audio_model_name = "Nidal/kashf-ai-audio-deepfake-detector"  # سيتم رفع النموذج لاحقًا على HuggingFace
feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_name)
audio_model = AutoModelForAudioClassification.from_pretrained(audio_model_name)
audio_classifier = pipeline("audio-classification", model=audio_model, feature_extractor=feature_extractor)

# ------------------- نموذج كشف التزييف في الفيديو / الوجه -------------------
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_audio_from_video(video_path):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(tmp_audio.name)
        return tmp_audio.name

def detect_deepfake_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fake_frames = 0
    total_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        # تحليل بسيط للـ artifacts (سيتم تحسينه لاحقًا بنموذج CNN أو Xception)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                blurred = cv2.GaussianBlur(face, (5,5), 0)
                edges = cv2.Canny(blurred, 100, 200)
                if np.mean(edges) < 20:  # عتبة بسيطة للكشف عن التمويه الزائد
                    fake_frames += 1
        total_frames += 1
    
    cap.release()
    fake_ratio = fake_frames / max(total_frames, 1)
    return fake_ratio

def analyze_file(file):
    if file is None:
        return "يرجى رفع ملف فيديو أو صوت", None
    
    result_text = ""
    
    # كشف الصوت
    if file.name.endswith(('.wav', '.mp3', '.m4a')):
        audio_results = audio_classifier(file.name)
        score = next(r['score'] for r in audio_results if r['label'] == "FAKE")
        result_text += f"🔊 الصوت: احتمالية التزييف {score*100:.2f}%\n"
        confidence = "مزيف" if score > 0.6 else "أصلي"
        result_text += f"النتيجة: {confidence}\n"
    
    # كشف الفيديو
    else:
        # استخراج الصوت أولًا
        audio_path = extract_audio_from_video(file.name)
        audio_results = audio_classifier(audio_path)
        audio_score = next(r['score'] for r in audio_results if r['label'] == "FAKE")
        
        # تحليل الفيديو
        video_score = detect_deepfake_video(file.name)
        
        result_text += f"🎥 الفيديو: احتمالية التزييف في الإطارات {video_score*100:.2
