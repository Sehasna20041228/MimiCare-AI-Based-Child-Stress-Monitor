# app_streamlit.py — Mimi AI Caregiver (Streamlit)
# Run: streamlit run app_streamlit.py
# Deploy: Docker SDK on Hugging Face Spaces (rename Dockerfile.streamlit -> Dockerfile)
# Needs: cv_core.py in same directory
#
# PACKAGES: streamlit numpy pandas Pillow opencv-python-headless ONLY
# NO: tensorflow pytorch deepface fer sklearn scipy gTTS joblib gradio
#
# VOICE: browser Web Speech API (window.speechSynthesis) — zero extra packages
# ANIMATION: pure CSS/SVG — zero extra packages

import sys
try:
    import pkg_resources
except ImportError:
    from importlib import resources as _r
    sys.modules["pkg_resources"] = _r

import os, base64, random, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as stc
from PIL import Image
from cv_core import analyse_photo, analyse_video

# ═══════════════════════════════════════════════
# PAGE SETUP
# ═══════════════════════════════════════════════
st.set_page_config(page_title="Mimi AI Caregiver", page_icon="🌟", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700;800&display=swap');
*{font-family:'Nunito',sans-serif}
body,.stApp{background:linear-gradient(135deg,#fef9ff 0%,#f0f4ff 50%,#fff0f9 100%)}
.disc{background:#fffbeb;border:2px solid #f59e0b;border-radius:14px;
  padding:12px 18px;margin:8px 0 14px;font-size:13px;color:#78350f;font-weight:700;line-height:1.6}
/* ── Mimi animated container ── */
.mw{display:flex;align-items:flex-end;gap:14px;margin:16px 0;
  animation:sli .5s cubic-bezier(.34,1.56,.64,1)}
@keyframes sli{from{opacity:0;transform:translateX(-30px)}to{opacity:1;transform:translateX(0)}}
.mc{flex-shrink:0}
/* Bounce animation on Mimi body */
.mimi-body{animation:bob 2s ease-in-out infinite}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
/* Mouth talking animation — toggled by JS */
.mimi-mouth-talk{animation:talk .25s steps(1) infinite}
@keyframes talk{0%{d:path("M38 60 Q50 72 62 60")}50%{d:path("M40 64 Q50 68 60 64")}}
.bub{background:#fff;border-radius:18px 18px 18px 4px;padding:12px 18px;font-size:15px;
  font-weight:600;color:#3d2c6b;line-height:1.5;border:2px solid #e8d8ff;max-width:460px;
  animation:pop .4s cubic-bezier(.34,1.56,.64,1)}
@keyframes pop{from{opacity:0;transform:scale(.87)}to{opacity:1;transform:scale(1)}}
/* Result cards */
.rc{border-radius:18px;padding:20px;margin:12px 0;text-align:center;font-weight:700;font-size:18px}
.rL{background:linear-gradient(135deg,#d4f5e2,#a8edca);color:#1a6640;border:2px solid #6fcea3}
.rM{background:linear-gradient(135deg,#fff3cd,#ffe082);color:#7a5400;border:2px solid #ffd54f}
.rH{background:linear-gradient(135deg,#ffd6d6,#ffaaaa);color:#8b1a1a;border:2px solid #ff7070}
/* Step dots */
.sb{display:flex;gap:8px;justify-content:center;margin:8px 0 18px}
.sd{width:11px;height:11px;border-radius:50%;background:#ddd;transition:background .3s}
.sd.active{background:#a855f7}.sd.done{background:#6fcea3}
/* Buttons */
.stButton>button{border-radius:50px!important;
  background:linear-gradient(135deg,#a855f7,#7c3aed)!important;
  color:#fff!important;font-weight:700!important;font-size:15px!important;
  padding:9px 28px!important;border:none!important}
/* Info boxes */
.tb{background:#f3e8ff;border-left:4px solid #a855f7;border-radius:0 10px 10px 0;
  padding:10px 16px;margin:7px 0;font-size:14px;color:#5b21b6}
.ob{background:#e0f2fe;border-left:4px solid #0284c7;border-radius:0 10px 10px 0;
  padding:10px 16px;margin:5px 0;font-size:14px;color:#0c4a6e}
.cv{background:#f0fdf4;border-left:4px solid #16a34a;border-radius:0 10px 10px 0;
  padding:10px 16px;margin:5px 0;font-size:14px;color:#14532d}
.st2{font-size:18px;font-weight:800;color:#4c1d95;margin:20px 0 6px;
  border-bottom:3px solid #e8d8ff;padding-bottom:4px}
/* Chat bubbles */
.chat-u{background:#e0f2fe;border-radius:12px 12px 4px 12px;
  padding:9px 14px;margin:5px 0 5px auto;max-width:78%;font-size:14px;
  font-weight:600;color:#0c4a6e;text-align:right}
.chat-m{background:#f3e8ff;border-radius:12px 12px 12px 4px;
  padding:9px 14px;margin:5px auto 5px 0;max-width:78%;font-size:14px;
  font-weight:600;color:#5b21b6}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# WEB SPEECH API — browser TTS, zero packages
# ═══════════════════════════════════════════════
def speak(text, rate=0.92, pitch=1.05):
    """
    Speaks text using the browser's built-in Web Speech API.
    No library, no internet request — runs entirely in the user's browser.
    Rate and pitch kept natural for a friendly assistant voice.
    Also triggers Mimi's mouth animation while speaking.
    """
    safe = text.replace("`","'").replace('"',"'").replace("\n"," ")
    js = f"""
    <script>
    (function(){{
      if(!window.speechSynthesis) return;
      window.speechSynthesis.cancel();
      var u = new SpeechSynthesisUtterance(`{safe}`);
      u.rate  = {rate};
      u.pitch = {pitch};
      u.lang  = 'en-US';
      // pick a female voice if available
      var voices = window.speechSynthesis.getVoices();
      var fem = voices.find(v => v.name.toLowerCase().includes('female') ||
                                  v.name.includes('Samantha') ||
                                  v.name.includes('Google UK English Female'));
      if(fem) u.voice = fem;
      // animate Mimi mouth while speaking
      var mouth = document.getElementById('mimi-mouth');
      if(mouth) mouth.classList.add('mimi-mouth-talk');
      u.onend = function(){{ if(mouth) mouth.classList.remove('mimi-mouth-talk'); }};
      window.speechSynthesis.speak(u);
    }})();
    </script>
    """
    stc.html(js, height=0)


# ═══════════════════════════════════════════════
# MIMI ANIMATED SVG CHARACTER
# ═══════════════════════════════════════════════
# Pure SVG + CSS — zero libraries
# The character has:
#   - Bouncing body (CSS @keyframes bob)
#   - Blinking eyes (CSS @keyframes blink)
#   - Talking mouth (CSS class toggled by JS when speaking)
#   - Expression variants: happy, thinking, excited, worried, calm

_BODY_SVG = """
<svg id="mimi-svg" width="110" height="130" viewBox="0 0 100 130"
     xmlns="http://www.w3.org/2000/svg">
  <style>
    .mimi-body {{ animation: bob 2s ease-in-out infinite; transform-origin: 50px 65px; }}
    @keyframes bob {{ 0%,100%{{transform:translateY(0)}} 50%{{transform:translateY(-8px)}} }}
    .mimi-eye  {{ animation: blink 4s ease-in-out infinite; transform-origin: center; }}
    @keyframes blink {{
      0%,45%,55%,100%{{transform:scaleY(1)}}
      48%,52%{{transform:scaleY(0.08)}}
    }}
  </style>
  <g class="mimi-body">
    <!-- Body / torso -->
    <rect x="28" y="90" width="44" height="38" rx="14" fill="#c084fc"/>
    <path d="M40 90 L50 105 L60 90" fill="#e9d5ff"/>
    <!-- Arms -->
    <ellipse cx="16" cy="105" rx="10" ry="7" fill="#c084fc" transform="rotate(-20,16,105)"/>
    <ellipse cx="84" cy="105" rx="10" ry="7" fill="#c084fc" transform="rotate(20,84,105)"/>
    <!-- Hands -->
    <circle cx="10" cy="112" r="7" fill="#fde68a"/>
    <circle cx="90" cy="112" r="7" fill="#fde68a"/>
    <!-- Legs -->
    <rect x="32" y="122" width="14" height="10" rx="6" fill="#7c3aed"/>
    <rect x="54" y="122" width="14" height="10" rx="6" fill="#7c3aed"/>
    <!-- Neck -->
    <rect x="43" y="80" width="14" height="12" rx="6" fill="#fde68a"/>
    <!-- Head -->
    <circle cx="50" cy="50" r="36" fill="#fde68a"/>
    <!-- Hair -->
    <ellipse cx="50" cy="16" rx="30" ry="14" fill="#92400e"/>
    <ellipse cx="20" cy="32" rx="10" ry="16" fill="#92400e"/>
    <ellipse cx="80" cy="32" rx="10" ry="16" fill="#92400e"/>
    <!-- Sparkle -->
    <text x="82" y="22" font-size="13">✨</text>
    <!-- Eyes (blinking) — replaced per expression below -->
    {eyes}
    <!-- Cheeks -->
    {cheeks}
    <!-- Mouth — id="mimi-mouth" so JS can animate it -->
    {mouth}
  </g>
</svg>"""

_EXPR = {
    "happy": {
        "eyes":   '<ellipse class="mimi-eye" cx="34" cy="46" rx="4" ry="5" fill="#3d2c6b"/>'
                  '<ellipse class="mimi-eye" cx="66" cy="46" rx="4" ry="5" fill="#3d2c6b"/>'
                  '<circle cx="35" cy="44" r="1.5" fill="white"/>'
                  '<circle cx="67" cy="44" r="1.5" fill="white"/>',
        "cheeks": '<ellipse cx="26" cy="56" rx="8" ry="5" fill="#ffb3c6" opacity=".65"/>'
                  '<ellipse cx="74" cy="56" rx="8" ry="5" fill="#ffb3c6" opacity=".65"/>',
        "mouth":  '<path id="mimi-mouth" d="M38 60 Q50 72 62 60" stroke="#3d2c6b"'
                  ' stroke-width="3" fill="none" stroke-linecap="round"/>',
    },
    "thinking": {
        "eyes":   '<ellipse class="mimi-eye" cx="34" cy="46" rx="4" ry="3" fill="#3d2c6b"/>'
                  '<ellipse class="mimi-eye" cx="66" cy="46" rx="4" ry="3" fill="#3d2c6b"/>'
                  '<circle cx="35" cy="45" r="1.2" fill="white"/>'
                  '<circle cx="67" cy="45" r="1.2" fill="white"/>',
        "cheeks": "",
        "mouth":  '<path id="mimi-mouth" d="M40 63 Q50 60 60 63" stroke="#3d2c6b"'
                  ' stroke-width="2.5" fill="none" stroke-linecap="round"/>'
                  '<circle cx="68" cy="44" r="3" fill="#fde68a" stroke="#3d2c6b" stroke-width="1"/>'
                  '<circle cx="72" cy="38" r="2" fill="#fde68a" stroke="#3d2c6b" stroke-width="1"/>'
                  '<circle cx="75" cy="33" r="1.5" fill="#fde68a" stroke="#3d2c6b" stroke-width="1"/>',
    },
    "excited": {
        "eyes":   '<ellipse class="mimi-eye" cx="34" cy="44" rx="5" ry="6" fill="#3d2c6b"/>'
                  '<ellipse class="mimi-eye" cx="66" cy="44" rx="5" ry="6" fill="#3d2c6b"/>'
                  '<circle cx="35.5" cy="42" r="2" fill="white"/>'
                  '<circle cx="67.5" cy="42" r="2" fill="white"/>',
        "cheeks": '<ellipse cx="24" cy="54" rx="9" ry="6" fill="#ffb3c6" opacity=".75"/>'
                  '<ellipse cx="76" cy="54" rx="9" ry="6" fill="#ffb3c6" opacity=".75"/>',
        "mouth":  '<path id="mimi-mouth" d="M34 60 Q50 76 66 60" stroke="#3d2c6b"'
                  ' stroke-width="3" fill="none" stroke-linecap="round"/>',
    },
    "worried": {
        "eyes":   '<ellipse class="mimi-eye" cx="34" cy="48" rx="4" ry="4" fill="#3d2c6b"/>'
                  '<ellipse class="mimi-eye" cx="66" cy="48" rx="4" ry="4" fill="#3d2c6b"/>'
                  '<line x1="28" y1="40" x2="40" y2="44" stroke="#3d2c6b" stroke-width="2"/>'
                  '<line x1="60" y1="44" x2="72" y2="40" stroke="#3d2c6b" stroke-width="2"/>',
        "cheeks": "",
        "mouth":  '<path id="mimi-mouth" d="M38 66 Q50 58 62 66" stroke="#3d2c6b"'
                  ' stroke-width="2.5" fill="none" stroke-linecap="round"/>',
    },
    "calm": {
        "eyes":   '<line x1="30" y1="46" x2="38" y2="46" stroke="#3d2c6b" stroke-width="3" stroke-linecap="round"/>'
                  '<line x1="62" y1="46" x2="70" y2="46" stroke="#3d2c6b" stroke-width="3" stroke-linecap="round"/>',
        "cheeks": '<ellipse cx="26" cy="54" rx="7" ry="4" fill="#ffb3c6" opacity=".5"/>'
                  '<ellipse cx="74" cy="54" rx="7" ry="4" fill="#ffb3c6" opacity=".5"/>',
        "mouth":  '<path id="mimi-mouth" d="M40 62 Q50 68 60 62" stroke="#3d2c6b"'
                  ' stroke-width="2.5" fill="none" stroke-linecap="round"/>',
    },
}

def show_mimi(msg, expr="happy", voice=True):
    """Render animated Mimi + speech bubble + optional TTS."""
    e = _EXPR.get(expr, _EXPR["happy"])
    svg = _BODY_SVG.format(**e)
    b64 = base64.b64encode(svg.encode()).decode()
    st.markdown(
        f'<div class="mw">'
        f'<div class="mc"><img src="data:image/svg+xml;base64,{b64}" width="110" height="130"/></div>'
        f'<div class="bub">{msg}</div></div>',
        unsafe_allow_html=True)
    if voice:
        # Strip HTML tags for TTS
        import re
        plain = re.sub(r"<[^>]+>", "", msg)
        speak(plain)


# ═══════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════
def disc():
    st.markdown(
        '<div class="disc">&#9888; <b>Important:</b> Mimi is a caregiver support tool — '
        'not a clinical or diagnostic tool. CV detects faces and basic image features only. '
        'Always consult a healthcare professional or autism specialist.</div>',
        unsafe_allow_html=True)

def sec(t):  st.markdown(f'<div class="st2">{t}</div>', unsafe_allow_html=True)
def tipb(t): st.markdown(f'<div class="tb">{t}</div>', unsafe_allow_html=True)
def obsb(t): st.markdown(f'<div class="ob">{t}</div>', unsafe_allow_html=True)
def cvb(t):  st.markdown(f'<div class="cv">{t}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# SCORING + RESULTS (shared with gradio via same logic)
# ═══════════════════════════════════════════════
_SM = {
    "sleep":    {"Much worse than usual":3,"Slightly worse":2,"About the same":0,"Better than usual":0},
    "comm":     {"Not communicating at all":3,"Much less than usual":2,"Slightly reduced":1,"About the same":0},
    "stim":     {"About the same":0,"Slightly more":1,"Significantly more":2,"Extremely intense / distressing":3},
    "eating":   {"Eating normally":0,"Slightly reduced":1,"Refusing some foods":2,"Refusing to eat":3},
    "sensory":  {"No more than usual":0,"Slightly more sensitive":1,"Noticeably more sensitive":2,"Covering ears / avoiding touch":3},
    "routine":  {"No disruption":0,"Minor change":1,"Moderate disruption":2,"Major disruption":3},
    "meltdown": {"No signs at all":0,"Mild signs — quieter or more rigid":1,
                 "Clear signs — crying, refusing, intense rocking":3,"Already in meltdown or shutdown":4},
    "new_beh":  {"No":0,"Minor — slightly different":1,"Yes — not seen before":2},
}
_LABS = {"sleep":"Sleep","comm":"Communication","stim":"Stimming","eating":"Eating",
         "sensory":"Sensory","routine":"Routine","meltdown":"Meltdown/shutdown","new_beh":"New behaviour"}

def score(ans): return sum(_SM[k].get(v,0) for k,v in ans.items())

def rdata(p):
    if p==0:
        return ("🎉 Great news! Your child appears <b>settled and regulated</b> today. "
                "Keep the routine consistent — you are doing a wonderful job! 💚",
                "excited","rL","🟢 Well Regulated — Child appears calm and settled",
                ["✅ Maintain today's routine as closely as possible",
                 "✅ Continue familiar, preferred activities",
                 "✅ Keep the environment calm and predictable",
                 "⏰ Check in again in 4–6 hours or if you notice changes"])
    if p==1:
        return ("🤔 I can see <b>some signs of stress or dysregulation</b> today. "
                "A little extra support and predictability will help a lot. 💛",
                "thinking","rM","🟡 Mild to Moderate Stress — Some dysregulation observed",
                ["💛 Offer a calm, low-stimulation space",
                 "💛 Stick to familiar routines — avoid new changes now",
                 "💛 Use your child's preferred calming strategies",
                 "💛 Speak in short, clear, predictable sentences",
                 "⏰ Check again within 1–2 hours"])
    return ("😟 Your child is showing <b>significant signs of stress or distress</b>. "
            "Please focus on their comfort and safety right now. You noticed this — that matters. ❤️",
            "worried","rH","🔴 High Stress / Distress — Immediate support needed",
            ["🚨 Stay close and stay calm yourself",
             "🚨 Reduce all sensory input: dim lights, lower noise, clear space",
             "🚨 Avoid demands or instructions during meltdown or shutdown",
             "🚨 Offer preferred comfort items silently",
             "🚨 Contact support team or GP if distress persists"])


# ═══════════════════════════════════════════════
# CHATBOT — keyword matching, zero ML
# ═══════════════════════════════════════════════
def chat_reply(msg):
    m = msg.lower()
    if any(w in m for w in ["meltdown","shutdown","crisis"]):
        return ("During a meltdown, reduce demands and sensory input immediately. "
                "Stay calm, speak as little as possible, and offer a safe space. "
                "Allow full recovery time before discussing what happened. 🛡️")
    if any(w in m for w in ["stim","stimming","rocking","flapping","spinning"]):
        return ("Stimming is natural self-regulation for autistic children — never try to "
                "eliminate it. Only redirect if it causes harm, and offer a safe alternative "
                "like a fidget toy or sensory object. 🌀")
    if any(w in m for w in ["sensory","noise","sound","light","texture","overwhelm"]):
        return ("Reduce stimulation: dim lights, noise-cancelling headphones, quiet room. "
                "A sensory diet of regular planned breaks prevents overload building up. 🎧")
    if any(w in m for w in ["routine","transition","change","schedule","unexpected"]):
        return ("Use visual schedules and give advance warnings before transitions. "
                "When unexpected changes happen, stay calm and validate their reaction — "
                "change is genuinely harder for autistic children. 📅")
    if any(w in m for w in ["communicate","communication","speech","nonverbal","aac","words"]):
        return ("Accept all forms of communication — pointing, gestures, AAC devices. "
                "Never pressure verbal responses during stress. Visual supports and "
                "picture cards can help greatly. 🗣️")
    if "sleep" in m:
        return ("A consistent, predictable bedtime routine helps most. Reduce screens "
                "1 hour before bed. Weighted blankets and white noise work well for many "
                "autistic children. Talk to your paediatrician if sleep problems are severe. 😴")
    if any(w in m for w in ["eat","food","meal","appetite","texture"]):
        return ("Food selectivity is very common in autism. Offer familiar safe foods alongside "
                "new ones without pressure. Never force eating. A dietitian experienced in "
                "autism can help with tailored guidance. 🍽️")
    if any(w in m for w in ["calm","calming","regulate","settle","regulation"]):
        return ("Calming strategies vary by child. Common approaches: weighted blanket, "
                "quiet safe space, preferred sensory items, or slow breathing modelled by you. "
                "Build a personalised calm-down toolkit when your child is already calm. 🌿")
    if any(w in m for w in ["anxious","anxiety","fear","scared","panic","worry"]):
        return ("Validate feelings first — 'I can see this feels scary for you.' "
                "Predictability and advance preparation reduce anxiety significantly. "
                "Social stories help prepare for specific situations. 💙")
    if any(w in m for w in ["social","friend","play","interact","eye contact"]):
        return ("Don't push eye contact — it is not a reliable sign of attention in autism. "
                "Allow parallel play rather than insisting on cooperative play. "
                "Follow their lead and join their interests. 🤝")
    if any(w in m for w in ["cv","photo","video","opencv","camera","detection"]):
        return ("The CV feature uses OpenCV's Haar Cascade face detector — it's built into "
                "the opencv library, so no download is needed. It then measures brightness "
                "(np.mean), contrast (np.std) and facial symmetry (left vs right pixel diff). "
                "No ML model or internet needed. 📷")
    if any(w in m for w in ["voice","speak","audio","sound","talking"]):
        return ("Mimi uses the browser's built-in Web Speech API — that's window.speechSynthesis "
                "in JavaScript. No library, no internet request. It runs entirely in your "
                "browser and works on every modern device. 🔊")
    if "help" in m:
        return ("I can help with: meltdowns, stimming, sensory overload, routines, "
                "communication, sleep, eating, calming strategies, anxiety, social interaction, "
                "and the CV and voice features. Just ask! 🌟")
    return ("Great question! Try asking about meltdowns, stimming, sensory needs, routines, "
            "sleep, eating, or calming strategies and I'll do my best to help. 😊")


# ═══════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════
_DEF = dict(step="welcome",pred=None,cl={},ph={},vi={},chat=[],mode="")
for k,v in _DEF.items():
    if k not in st.session_state: st.session_state[k]=v

def reset():
    for k,v in _DEF.items(): st.session_state[k]=v

char = st.sidebar.text_input("Assistant name", value="Mimi")
mute = st.sidebar.checkbox("🔇 Mute voice", value=False)

# Step indicator dots
_STEPS = ["welcome","mode","analyze","result","chat"]
si   = _STEPS.index(st.session_state.step) if st.session_state.step in _STEPS else 0
dots = "".join(
    f'<div class="sd {"done" if i<si else ("active" if i==si else "")}"></div>'
    for i in range(len(_STEPS)))
st.markdown(f'<div class="sb">{dots}</div>', unsafe_allow_html=True)
st.markdown(
    f"<h1 style='text-align:center;color:#4c1d95;font-size:1.85rem;margin-bottom:0'>"
    f"🌟 {char} AI Caregiver</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#7c3aed;font-weight:600;margin-top:2px'>"
    "Supporting autistic children's emotional wellbeing</p>", unsafe_allow_html=True)

use_voice = not mute  # sidebar mute toggle


# ════════════════════════════════════════════════
# STEP 1 — WELCOME
# ════════════════════════════════════════════════
if st.session_state.step == "welcome":
    disc()
    show_mimi(
        f"Hi there! 👋 I'm <b>{char}</b>, your friendly caregiver assistant! "
        "I'll help you observe how your child is doing today using a behaviour "
        "checklist, optional photo or video analysis, and a chatbot for advice. "
        "Let's begin! 🌈",
        "excited", use_voice)
    st.markdown("<br>", unsafe_allow_html=True)
    _,c2,_ = st.columns([1,2,1])
    with c2:
        if st.button("Let's Begin! 🚀"):
            st.session_state.step = "mode"; st.rerun()


# ════════════════════════════════════════════════
# STEP 2 — MODE
# ════════════════════════════════════════════════
elif st.session_state.step == "mode":
    show_mimi("Choose how you'd like to assess your child today! 😊",
              "thinking", use_voice)
    disc()
    sec("🧭 Choose Assessment Mode")
    mode = st.radio("", [
        "📋 Behaviour checklist only",
        "📷 Photo CV only",
        "🎥 Video CV only",
        "📋 + 📷 Checklist and photo",
        "📋 + 🎥 Checklist and video",
        "📋 + 📷 + 🎥 All three",
    ], label_visibility="collapsed")
    tipb("💡 <b>CV method:</b> Haar Cascade face detection + brightness / contrast / symmetry. "
         "Lightweight — no internet, no ML model needed.")
    _,c2,_ = st.columns([1,2,1])
    with c2:
        if st.button("Next ➡️"):
            st.session_state.mode = mode
            st.session_state.step = "analyze"
            st.rerun()


# ════════════════════════════════════════════════
# STEP 3 — ANALYSIS
# ════════════════════════════════════════════════
elif st.session_state.step == "analyze":
    mode = st.session_state.mode
    disc()
    cl_s=0; cl_a={}; pr={}; vr={}

    # ── Behaviour checklist ──────────────────────
    if "checklist" in mode.lower() or "all" in mode.lower():
        show_mimi(
            "I'll ask you questions about your child's behaviour today. "
            "Answer based on what you've observed — you know your child best! 💛",
            "happy", use_voice)
        sec("📋 Behaviour Observation Checklist")
        st.caption("Compare to your child's usual baseline.")
        sl = st.select_slider("Sleep last night?",
            ["Much worse than usual","Slightly worse","About the same","Better than usual"],
            value="About the same", key="qs")
        co = st.select_slider("Communication today?",
            ["Not communicating at all","Much less than usual","Slightly reduced","About the same"],
            value="About the same", key="qc")
        st_ = st.select_slider("Stimming vs usual?",
            ["About the same","Slightly more","Significantly more","Extremely intense / distressing"],
            value="About the same", key="qt")
        ea = st.select_slider("Eating today?",
            ["Eating normally","Slightly reduced","Refusing some foods","Refusing to eat"],
            value="Eating normally", key="qe")
        se = st.select_slider("Sensory sensitivity?",
            ["No more than usual","Slightly more sensitive","Noticeably more sensitive","Covering ears / avoiding touch"],
            value="No more than usual", key="qse")
        ro = st.select_slider("Routine disruption?",
            ["No disruption","Minor change","Moderate disruption","Major disruption"],
            value="No disruption", key="qr")
        ml = st.radio("Meltdown / shutdown signs?",
            ["No signs at all","Mild signs — quieter or more rigid",
             "Clear signs — crying, refusing, intense rocking","Already in meltdown or shutdown"],
            key="qm")
        nb = st.radio("New or unusual behaviour?",
            ["No","Minor — slightly different","Yes — not seen before"], key="qn")
        cl_a = dict(sleep=sl,comm=co,stim=st_,eating=ea,
                    sensory=se,routine=ro,meltdown=ml,new_beh=nb)
        cl_s = score(cl_a)

    # ── Photo CV ─────────────────────────────────
    if "photo" in mode.lower() or "all" in mode.lower():
        sec("📷 Photo Analysis (CV)")
        show_mimi("Upload a clear photo — I'll detect the face and analyse "
                  "brightness, contrast and symmetry. 📸", "thinking", use_voice)
        tipb("📚 <b>Method:</b> Haar Cascade → face crop → "
             "np.mean (brightness) · np.std (contrast) · L/R pixel diff (symmetry)")
        up = st.file_uploader("Upload photo (JPG/PNG)",
                              type=["jpg","jpeg","png"], key="ph")
        if up:
            pil = Image.open(up).convert("RGB")
            with st.spinner("Running photo CV..."):
                pr, ann = analyse_photo(pil)
            ca,cb = st.columns(2)
            with ca: st.image(np.array(pil), caption="Original", use_column_width=True)
            with cb: st.image(ann, caption="CV output — face detection", use_column_width=True)
            sec("🔬 Photo CV Findings")
            if pr["face_detected"]:
                cvb(f"✅ Face detected — {pr['face_count']} face(s)")
                cvb(f"Brightness: {pr['brightness']}/255 | Contrast: {pr['contrast']} | "
                    f"Symmetry diff: {pr['symmetry_score']}")
                for o in pr["observations"]: cvb(f"• {o}")
            else:
                st.warning("No face detected — try a clearer, well-lit photo.")

    # ── Video CV ─────────────────────────────────
    if "video" in mode.lower() or "all" in mode.lower():
        sec("🎥 Video Analysis (CV)")
        show_mimi("Upload a short MP4 — I'll sample frames and measure "
                  "brightness, contrast and symmetry over time. 🎥", "thinking", use_voice)
        tipb("📚 <b>Method:</b> VideoCapture → sample every 15th frame → "
             "Haar Cascade per frame → aggregate stats. ~50 frames for a 30s video.")
        vup = st.file_uploader("Upload video (MP4, ≤30s recommended)",
                               type=["mp4"], key="vid")
        if vup:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(vup.read()); tp = tmp.name
            try:
                with st.spinner("Sampling frames and running CV..."):
                    vr, fst, sfr = analyse_video(tp, sample_every=15, max_frames=60)
                sec("🔬 Video CV Findings")
                for o in vr["observations"]: cvb(f"• {o}")
                if vr["avg_brightness"] is not None:
                    cvb(f"Avg brightness: {vr['avg_brightness']}/255 | "
                        f"Avg contrast: {vr['avg_contrast']} | "
                        f"Avg symmetry diff: {vr['avg_symmetry']}")
                if sfr:
                    st.markdown("**Sample annotated frames:**")
                    cols = st.columns(len(sfr))
                    for col, frm in zip(cols, sfr):
                        with col: st.image(frm, use_column_width=True)
                if fst:
                    bd = [{"Time(s)":s["time_s"],"Brightness":s["brightness"]}
                          for s in fst if s["brightness"] is not None]
                    if bd:
                        st.markdown("**Brightness over time:**")
                        st.line_chart(pd.DataFrame(bd).set_index("Time(s)"))
            finally:
                try: os.unlink(tp)
                except OSError: pass

    # ── Submit ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _,c2,_ = st.columns([1,2,1])
    with c2:
        if st.button("See Results 🔍"):
            total = cl_s + pr.get("cv_score",0) + vr.get("cv_score",0)
            pred  = 0 if total<=4 else (1 if total<=13 else 2)
            st.session_state.update(pred=int(pred), cl=cl_a, ph=dict(pr), vi=dict(vr), step="result")
            st.rerun()


# ════════════════════════════════════════════════
# STEP 4 — RESULTS
# ════════════════════════════════════════════════
elif st.session_state.step == "result":
    disc()
    pred = st.session_state.pred
    msg, expr, card, label, tips = rdata(pred)

    show_mimi(msg, expr, use_voice)
    st.markdown(f'<div class="rc {card}">{label}</div>', unsafe_allow_html=True)

    # Behaviour observations
    if st.session_state.cl:
        sec("📋 Behaviour Observations")
        for k,v in st.session_state.cl.items():
            obsb(f"• <b>{_LABS.get(k,k)}:</b> {v}")

    # Photo CV summary
    p = st.session_state.ph
    if p.get("face_detected"):
        sec("📷 Photo CV Summary")
        cvb(f"• Faces: {p['face_count']} | Brightness: {p['brightness']} | "
            f"Contrast: {p['contrast']} | Symmetry: {p['symmetry_score']}")
        for o in p["observations"]: cvb(f"• {o}")

    # Video CV summary
    v = st.session_state.vi
    if v.get("with_face",0) > 0:
        sec("🎥 Video CV Summary")
        for o in v["observations"]: cvb(f"• {o}")

    # Recommendations
    sec("💡 Recommended Actions")
    for t in tips: tipb(t)

    # Trend chart
    sec("📈 Observation Trend")
    st.caption("'Previous' is illustrative — data is not stored between sessions.")
    prev = min(2, max(0, pred + random.choice([-1,0,1])))
    st.line_chart(pd.DataFrame(
        {"Check":["Previous","Now"],"Stress Level":[prev,pred]}
    ).set_index("Check"))

    # ── OPTIONAL CHATBOT after results ────────────
    st.markdown("<br>", unsafe_allow_html=True)
    sec("💬 Ask Mimi — Optional Support Chat")
    show_mimi(
        "Any questions about what to do next? Ask me about meltdowns, sensory needs, "
        "routines, stimming, calming strategies and more! 😊",
        "happy", use_voice)

    # Chat history display
    for spk, txt in st.session_state.chat:
        if spk == "You":
            st.markdown(f'<div class="chat-u">🧑 {txt}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-m">🌟 <b>{char}:</b> {txt}</div>', unsafe_allow_html=True)

    ui = st.text_input("Type your question:", key="ci",
                       placeholder="e.g. How do I help during a meltdown?")
    c1,c2,c3 = st.columns([1,1,1])
    with c1:
        if st.button("Send 💬"):
            if ui.strip():
                reply = chat_reply(ui)
                st.session_state.chat.extend([("You",ui),(char,reply)])
                if use_voice: speak(reply)
                st.rerun()
    with c2:
        if st.button("Clear chat 🗑️"):
            st.session_state.chat = []; st.rerun()
    with c3:
        if st.button("🔄 Start Over"):
            reset(); st.rerun()


# ════════════════════════════════════════════════
# STEP 5 — DEDICATED CHAT PAGE (from nav)
# ════════════════════════════════════════════════
elif st.session_state.step == "chat":
    show_mimi("Ask me anything about supporting your child — I'm always here! 💬✨",
              "happy", use_voice)
    disc()
    sec("💬 Chat with Mimi")
    for spk, txt in st.session_state.chat:
        if spk == "You":
            st.markdown(f'<div class="chat-u">🧑 {txt}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-m">🌟 <b>{char}:</b> {txt}</div>', unsafe_allow_html=True)
    ui = st.text_input("Ask me something:", key="ci2",
                       placeholder="e.g. How do I help during a meltdown?")
    c1,_,c3 = st.columns([1,1,1])
    with c1:
        if st.button("Send 💬"):
            if ui.strip():
                reply = chat_reply(ui)
                st.session_state.chat.extend([("You",ui),(char,reply)])
                if use_voice: speak(reply)
                st.rerun()
    with c3:
        if st.button("🔄 Start Over"): reset(); st.rerun()
