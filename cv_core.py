# cv_core.py — Computer Vision core for Mimi AI Caregiver
# Uses OpenCV Haar Cascade ONLY — no ML models, no internet, lightweight
# Returns plain Python dicts (int/float/str/list) — safe for st.session_state

import cv2
import numpy as np
from PIL import Image


# ── Load Haar Cascade once at import time ──────────────────────────────────
_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR array."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR array to RGB for Streamlit display."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _detect_faces(gray: np.ndarray):
    """Run Haar Cascade face detection. Returns list of (x,y,w,h)."""
    faces = _CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return []
    return faces.tolist()


def _brightness(gray: np.ndarray) -> float:
    return round(float(np.mean(gray)), 1)


def _contrast(gray: np.ndarray) -> float:
    return round(float(np.std(gray)), 1)


def _symmetry(gray: np.ndarray) -> float:
    """Left-vs-right pixel difference as a symmetry proxy."""
    h, w = gray.shape
    left  = gray[:, : w // 2]
    right = np.fliplr(gray[:, w - w // 2 :])
    min_w = min(left.shape[1], right.shape[1])
    diff  = np.mean(np.abs(left[:, :min_w].astype(float) - right[:, :min_w].astype(float)))
    return round(float(diff), 2)


def _cv_score_from_brightness(brightness: float) -> int:
    """
    Simple heuristic: very dark or very bright images score higher stress.
    Returns 0–3 to feed into the total stress score.
    """
    if brightness < 60 or brightness > 220:
        return 2
    if brightness < 90 or brightness > 190:
        return 1
    return 0


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC API — analyse_photo
# ══════════════════════════════════════════════════════════════════════════

def analyse_photo(pil_img: Image.Image):
    """
    Analyse a PIL image for face detection and basic CV metrics.

    Returns
    -------
    result : dict  — plain Python types, safe for st.session_state
    annotated : np.ndarray (H×W×3, uint8, RGB) — for st.image display
    """
    bgr  = _pil_to_bgr(pil_img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    faces        = _detect_faces(gray)
    face_count   = len(faces)
    face_detected = face_count > 0

    brightness   = _brightness(gray)
    contrast     = _contrast(gray)
    symmetry     = _symmetry(gray)
    cv_score     = _cv_score_from_brightness(brightness)

    observations = []
    if not face_detected:
        observations.append("No face detected — ensure the photo is well-lit and the face is visible.")
        cv_score = 0
    else:
        observations.append(f"{face_count} face(s) detected successfully.")
        if brightness < 80:
            observations.append("Image is quite dark — lighting may affect accuracy.")
            cv_score = max(cv_score, 2)
        elif brightness > 200:
            observations.append("Image is very bright / overexposed.")
            cv_score = max(cv_score, 1)
        else:
            observations.append("Lighting appears adequate for analysis.")
        if contrast < 20:
            observations.append("Low contrast — image may be blurry or washed out.")
        if symmetry > 30:
            observations.append("Higher facial asymmetry detected — may reflect posture or expression.")
        elif symmetry < 10:
            observations.append("Facial symmetry appears typical.")

    # Draw bounding boxes on annotated copy
    annotated = bgr.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (109, 40, 217), 2)
        cv2.putText(annotated, "Face", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (109, 40, 217), 2)

    result = dict(
        face_detected = bool(face_detected),
        face_count    = int(face_count),
        brightness    = float(brightness),
        contrast      = float(contrast),
        symmetry_score= float(symmetry),
        cv_score      = int(cv_score),
        observations  = [str(o) for o in observations],
    )

    return result, _bgr_to_rgb(annotated)


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC API — analyse_video
# ══════════════════════════════════════════════════════════════════════════

def analyse_video(video_path: str, sample_every: int = 15, max_frames: int = 60):
    """
    Sample frames from a video file and compute per-frame CV metrics.

    Parameters
    ----------
    video_path   : path to the temp MP4 file
    sample_every : take one frame every N frames
    max_frames   : maximum number of sampled frames to process

    Returns
    -------
    result  : dict  — plain Python types, safe for st.session_state
    frame_stats : list of dicts (one per sampled frame)
    sample_frames : list of np.ndarray (RGB) for display (up to 4)
    """
    cap = cv2.VideoCapture(video_path)

    frame_stats   = []
    sample_frames = []
    frame_idx     = 0
    processed     = 0
    with_face     = 0

    while cap.isOpened() and processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces  = _detect_faces(gray)
            br     = _brightness(gray)
            co     = _contrast(gray)
            sym    = _symmetry(gray)
            fps    = cap.get(cv2.CAP_PROP_FPS) or 25
            time_s = round(frame_idx / fps, 2)

            if faces:
                with_face += 1
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (109, 40, 217), 2)

            frame_stats.append(dict(
                frame     = int(frame_idx),
                time_s    = float(time_s),
                brightness= float(br),
                contrast  = float(co),
                symmetry  = float(sym),
                face_count= int(len(faces)),
            ))

            # Keep up to 4 sample frames for display
            if len(sample_frames) < 4:
                sample_frames.append(_bgr_to_rgb(frame))

            processed += 1

        frame_idx += 1

    cap.release()

    # Aggregate stats
    brightnesses = [f["brightness"] for f in frame_stats if f["brightness"] is not None]
    contrasts    = [f["contrast"]   for f in frame_stats if f["contrast"]   is not None]
    symmetries   = [f["symmetry"]   for f in frame_stats if f["symmetry"]   is not None]

    avg_br  = round(float(np.mean(brightnesses)), 1) if brightnesses else None
    avg_co  = round(float(np.mean(contrasts)),    1) if contrasts    else None
    avg_sym = round(float(np.mean(symmetries)),   2) if symmetries   else None

    # Build observations
    observations = []
    total = len(frame_stats)
    if total == 0:
        observations.append("No frames could be read from the video.")
        cv_score = 0
    else:
        observations.append(f"{total} frames sampled — {with_face} contained a detected face.")
        if with_face == 0:
            observations.append("No face detected in any frame — try a shorter, well-lit clip.")
            cv_score = 0
        else:
            face_ratio = with_face / total
            if face_ratio < 0.3:
                observations.append("Face visible in fewer than 30% of frames — child may be moving a lot.")
                cv_score = 2
            elif face_ratio < 0.6:
                observations.append("Face visible in about half of frames — moderate movement detected.")
                cv_score = 1
            else:
                observations.append("Face consistently visible across frames.")
                cv_score = 0

            if avg_br is not None:
                if avg_br < 70:
                    observations.append("Video appears quite dark overall.")
                    cv_score = max(cv_score, 1)
                elif avg_br > 210:
                    observations.append("Video appears overexposed.")
                elif 90 <= avg_br <= 190:
                    observations.append("Lighting across the video appears good.")

            if avg_sym is not None and avg_sym > 35:
                observations.append("Higher average facial asymmetry — may reflect movement or expression changes.")

    result = dict(
        total_frames_sampled = int(total),
        with_face            = int(with_face),
        avg_brightness       = avg_br,
        avg_contrast         = avg_co,
        avg_symmetry         = avg_sym,
        cv_score             = int(cv_score),
        observations         = [str(o) for o in observations],
    )

    return result, frame_stats, sample_frames
