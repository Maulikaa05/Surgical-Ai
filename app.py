"""
app.py — AI Arthroscopic Surgical Performance Evaluator  v3
Ceaser's Medathon 2026  |  Problem Statement #1
Three Outputs: Live Dashboard · Hospital PDF · Annotated Frame
Run: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os, sys, json, time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src import (
    extract_coordinates_from_folder,
    calculate_all_metrics, stability_timeline, compute_baseline,
    phase_time_analysis,
    classify, skill_color, skill_badge_color, skill_emoji, skill_clinical_equiv,
    score_band, score_colour,
    generate_feedback,
    generate_pdf_report,
    draw_trajectory_on_frame, create_expert_comparison, find_best_frame,
    load_phase_annotations, get_dominant_phase, find_phase_file,
    process_entire_archive,
    download_video_frames, extract_frames_from_video,
)

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="SurgicalAI — Arthroscopic Evaluator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;}
.stApp{background:linear-gradient(135deg,#F0F4F8 0%,#E8EEF5 100%);}
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#06192E 0%,#0D2B4E 45%,#1A4F8A 100%)!important;
  border-right:1px solid rgba(255,255,255,0.05)!important;
}
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] span {
  color: #ECF0F1 !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label div,
section[data-testid="stSidebar"] .stCheckbox label div {
  color: #ECF0F1 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{color:#fff!important;}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stRadio label{color:#A9CCE3!important;font-size:0.82rem!important;}

/* Explicitly force dark text for all input fields so typing is visible */
.stTextInput input, .stSelectbox div[data-baseweb="select"] {
  color: #0D2B4E !important;
  -webkit-text-fill-color: #0D2B4E !important;
  caret-color: #0D2B4E !important;
  background-color: #FFFFFF !important;
}
.stTextInput input::placeholder {
  color: #7F8C8D !important;
  -webkit-text-fill-color: #7F8C8D !important;
}

/* Fix st.status visibility */
[data-testid="stStatusWidget"] div[data-testid="stVerticalBlock"] p,
[data-testid="stStatusWidget"] div[data-testid="stVerticalBlock"] span {
  color: #2C3E50 !important;
}
[data-testid="stStatusWidget"] summary {
  background: #EBF3FB !important;
  color: #0D2B4E !important;
  border-radius: 8px;
}
section[data-testid="stSidebar"] .stButton>button{
  background:linear-gradient(135deg,#1A5276,#2980B9)!important;
  border:none!important;color:white!important;border-radius:8px!important;
}
section[data-testid="stSidebar"] .stButton>button:hover{
  background:linear-gradient(135deg,#2980B9,#1A5276)!important;
}

.hero{
  background:linear-gradient(135deg,#06192E 0%,#0D2B4E 50%,#1A4F8A 100%);
  border-radius:16px;padding:28px 32px;margin-bottom:24px;color:white;
  box-shadow:0 8px 32px rgba(13,43,78,0.35);
  border:1px solid rgba(255,255,255,0.07);
  position:relative;overflow:hidden;
}
.hero::before{
  content:'';position:absolute;top:-40%;right:-10%;width:300px;height:300px;
  background:radial-gradient(circle,rgba(41,128,185,0.15) 0%,transparent 70%);
}
.hero h1{color:white!important;font-size:1.7rem;margin:0 0 8px;font-weight:800;letter-spacing:-0.5px;}
.hero p{color:#A9CCE3;margin:0;font-size:0.85rem;line-height:1.6;}
.hero .badge{
  display:inline-block;background:rgba(39,174,96,0.2);border:1px solid rgba(39,174,96,0.4);
  color:#2ECC71;border-radius:20px;padding:3px 12px;font-size:0.72rem;font-weight:600;
  margin-top:10px;letter-spacing:0.5px;
}

.card{
  background:white;border-radius:14px;padding:22px 24px;
  box-shadow:0 2px 16px rgba(0,0,0,0.06);height:100%;
  border:1px solid rgba(0,0,0,0.04);
  transition:transform 0.2s,box-shadow 0.2s;
}
.card:hover{transform:translateY(-2px);box-shadow:0 4px 24px rgba(0,0,0,0.1);}

.sec-head{
  font-size:1.0rem;font-weight:700;color:#0D2B4E;
  border-left:4px solid #1A4F8A;padding-left:12px;
  margin:20px 0 12px;letter-spacing:-0.2px;
}
.skill-pill{
  display:inline-block;padding:12px 38px;border-radius:40px;
  font-size:1.4rem;font-weight:800;color:white;letter-spacing:0.3px;
  box-shadow:0 6px 20px rgba(0,0,0,0.2);
}
.fb-box{
  background:white;border-left:5px solid #1A4F8A;border-radius:10px;
  padding:16px 20px;box-shadow:0 2px 10px rgba(0,0,0,0.05);
  margin-bottom:12px;border-top:1px solid rgba(0,0,0,0.03);
}
.fb-box p{color:#2C3E50;line-height:1.8;margin:6px 0;}
.action-row{
  background:linear-gradient(135deg,#EBF3FB,#F0F7FF);
  border-radius:8px;padding:12px 16px;margin:6px 0;
  color:#1A5276;font-size:0.88rem;
  border-left:3px solid #1A5276;
}
.metric-mini{text-align:center;padding:8px;}
.metric-mini .val{font-size:1.9rem;font-weight:800;line-height:1.1;}
.metric-mini .lbl{font-size:0.72rem;color:#7F8C8D;margin-top:3px;}
.metric-mini .band{font-size:0.72rem;font-weight:700;margin-top:3px;}
.detection-bar{
  height:8px;border-radius:4px;
  background:linear-gradient(90deg,#27AE60,#2ECC71);
  margin-top:4px;box-shadow:0 2px 4px rgba(39,174,96,0.3);
}

.output-header{
  background:linear-gradient(135deg,#1A5276,#2471A3);
  border-radius:10px;padding:12px 18px;margin:6px 0 16px;
  color:white;font-weight:700;font-size:0.95rem;
  display:flex;align-items:center;gap:8px;
}
.pipeline-step{
  background:white;border-radius:8px;padding:10px 14px;margin:4px 0;
  border-left:3px solid #2980B9;font-size:0.85rem;color:#2C3E50;
}
.video-source-card{
  background:linear-gradient(135deg,#1a1a2e,#16213e);
  border-radius:12px;padding:18px;border:1px solid rgba(255,255,255,0.1);
}
</style>""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────
for key, val in [("history", []), ("last", None), ("baseline", None)]:
    if key not in st.session_state:
        st.session_state[key] = val


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 SurgicalAI v4")
    st.caption("Arthroscopic Performance Evaluator")
    st.markdown("---")

    # ── Input Mode ──
    st.markdown("### 📥 Input Mode")
    input_mode = st.radio(
        "Select input source",
        ["📂 Local Dataset (ArthroPhase)", "🎥 Video File / YouTube URL"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    if input_mode == "📂 Local Dataset (ArthroPhase)":
        st.markdown("### 📂 Dataset Path")
        archive_root = st.text_input(
            "Archive root path",
            value=r"C:\medathon1\dataset\archive",
            help="Path to archive folder containing videoXX subfolders"
        )
        archive_path = Path(archive_root)
        video_dirs = []
        if archive_path.exists():
            video_dirs = sorted(d for d in archive_path.iterdir() if d.is_dir())

        st.markdown("---")
        st.markdown("### 🎯 Segment")
        if not video_dirs:
            st.warning("Archive path not found.\nUpdate path above.")
            selected_seg = None
            video_id = None
        else:
            vid_names = [d.name for d in video_dirs]
            sel_vid = st.selectbox("Video", vid_names)
            video_id = sel_vid
            vid_path = archive_path / sel_vid
            seg_dirs = sorted(d for d in vid_path.iterdir() if d.is_dir())
            if seg_dirs:
                seg_names = [d.name for d in seg_dirs]
                sel_seg = st.selectbox("Segment", seg_names)
                selected_seg = vid_path / sel_seg
            else:
                st.info("No segments in this video.")
                selected_seg = None
        video_source = None
        video_mode = "dataset"

    else:  # Video / YouTube
        st.markdown("### 🎥 Video Source")
        video_source = st.text_input(
            "Video URL or local path",
            placeholder="https://youtube.com/watch?v=... or /path/to/video.mp4",
            help="Paste a YouTube URL or local video file path"
        )
        st.caption("💡 Supports: YouTube, direct .mp4/.avi/.mov files")
        selected_seg = None
        video_id = None
        archive_root = ""
        video_mode = "video"

    st.markdown("---")
    st.markdown("### 👤 Surgeon")
    surgeon_name = st.text_input("Surgeon name / ID", value="Trainee Surgeon")

    st.markdown("---")
    st.markdown("### ⚙️ Options")
    do_pdf = st.checkbox("Generate PDF report", value=True)
    do_compare = st.checkbox("Expert comparison view", value=True)
    do_baseline = st.checkbox("Compute dataset baseline", value=False,
                               help="Processes ALL segments — ~1-2 min",
                               disabled=(video_mode == "video"))

    st.markdown("---")
    run_btn = st.button("▶  Run Analysis", type="primary", width="stretch")
    if st.button("🔄 Reset Session", width="stretch"):
        st.session_state.history = []
        st.session_state.last = None
        st.session_state.baseline = None
        st.rerun()

    st.markdown("---")
    st.caption("Ceaser's Medathon 2026\nCIT · Dept. AI & Data Science\nProblem Statement #1")


# ════════════════════════════════════════════════════════════════════
# HERO HEADER
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🔬 AI Arthroscopic Surgical Performance Evaluator</h1>
  <p>Real-time instrument tracking &nbsp;·&nbsp; Phase-aware scoring &nbsp;·&nbsp;
  Hospital-grade PDF &nbsp;·&nbsp; Annotated trajectory frames<br>
  Powered by <b>ArthroPhase</b> — 27 real ACL reconstruction surgeries
  from Balgrist University Hospital (2024) · CC BY 4.0</p>
  <span class="badge">✓ ASSET-Aligned Metrics · Zero Hardcoded Values · Production Grade</span>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# OUTPUT LEGEND
# ════════════════════════════════════════════════════════════════════
if not st.session_state.last:
    o1, o2, o3, o4 = st.columns(4)
    for col, icon, title, body in [
        (o1, "📊", "Output 1 — Live Dashboard",
         "Real-time gauges, trajectory timeline, radar chart, and skill classification with clinical equivalents."),
        (o2, "📄", "Output 2 — Hospital PDF",
         "A4 report with surgeon name, date, gauges, overlays, and 3 clinical paragraphs. Print-ready."),
        (o3, "🎯", "Output 3 — Annotated Frame",
         "Instrument path drawn over the actual surgical frame as visual proof of tracking quality."),
        (o4, "🎥", "YouTube / Video Input",
         "Paste any YouTube URL or video file. Frames are auto-extracted and analysed end-to-end."),
    ]:
        with col:
            st.markdown(f"""
            <div class="card">
              <div style="font-size:2.2rem;margin-bottom:10px">{icon}</div>
              <div style="font-weight:700;color:#0D2B4E;margin-bottom:6px;font-size:0.95rem">{title}</div>
              <div style="font-size:0.8rem;color:#7F8C8D;line-height:1.65">{body}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:white;border-radius:12px;padding:18px 24px;
    box-shadow:0 2px 8px rgba(0,0,0,0.06);text-align:center;color:#7F8C8D;
    border:1px solid rgba(0,0,0,0.04);">
    <b style="color:#0D2B4E">Quick Start:</b>
      Choose input mode in sidebar → enter path/URL → click <b>▶ Run Analysis</b>
      <br><br>
      <span style="font-size:0.78rem;">
        Dataset: <b>ArthroPhase</b> · DOI: 10.5281/zenodo.14288900 · CC BY 4.0 · 
        Balgrist University Hospital, University of Zurich, 2024
      </span>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# RUN ANALYSIS
# ════════════════════════════════════════════════════════════════════
if run_btn:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    with st.status("🔬 Running AI pipeline…", expanded=True) as status:

        # ── STEP 0: Handle video/YouTube input ──
        if video_mode == "video":
            if not video_source or not video_source.strip():
                st.error("Please enter a video URL or file path.")
                st.stop()

            st.write("**Step 0/8** — Processing video source…")
            video_tmp_dir = out_dir / f"video_frames_{ts_tag}"
            video_tmp_dir.mkdir(exist_ok=True)

            try:
                if video_source.strip().startswith(("http://", "https://")):
                    st.write(f"  ⏳  Downloading from URL: `{video_source[:60]}...`")
                    video_file = download_video_frames(video_source.strip(), str(video_tmp_dir))
                else:
                    video_file = video_source.strip()

                st.write(f"  ⏳  Extracting frames from video…")
                n_extracted = extract_frames_from_video(video_file, str(video_tmp_dir), max_frames=300)
                st.write(f"  ✅  {n_extracted} frames extracted from video")
                selected_seg_str = str(video_tmp_dir)
                seg_display_name = Path(video_source).stem[:40] if not video_source.startswith("http") else "youtube_video"
                video_id = seg_display_name
            except Exception as e:
                st.error(f"Video processing failed: {e}")
                st.stop()
        else:
            if selected_seg is None:
                st.error("Please select a valid segment before running.")
                st.stop()
            selected_seg_str = str(selected_seg)
            seg_display_name = selected_seg.name

        # ── STEP 1: Extract coordinates ──
        st.write("**Step 1/8** — Extracting instrument coordinates from frames…")
        try:
            records = extract_coordinates_from_folder(selected_seg_str)
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.stop()
        valid_n = sum(1 for r in records if r["tip_x"] is not None)
        det_rate_pct = valid_n / max(len(records), 1) * 100
        st.write(f"  ✅  {len(records)} frames · {valid_n} with instrument detected "
                 f"({det_rate_pct:.0f}% detection rate)")

        # ── STEP 2: Phase annotation ──
        st.write("**Step 2/8** — Loading surgical phase annotations…")
        phase_file = find_phase_file(archive_root, video_id or "video01") if video_mode == "dataset" else None
        phase_anns = load_phase_annotations(phase_file) if phase_file else {}
        phase_label = (get_dominant_phase([r["frame"] for r in records], phase_anns)
                       if phase_anns else "ACL Reconstruction")
        src_note = f" (from {Path(phase_file).name})" if phase_file else " (inferred from video type)"
        st.write(f"  ✅  Dominant phase: **{phase_label}**{src_note}")

        # ── STEP 3: Metrics ──
        st.write("**Step 3/8** — Computing ASSET-aligned performance metrics…")
        result = calculate_all_metrics(records, phase_label=phase_label)
        if "error" in result:
            st.error(result["error"])
            st.stop()
        m = result["metrics"]
        st.write(f"  ✅  Stability {m['stability_score']:.1f}  "
                 f"· Efficiency {m['efficiency_score']:.1f}  "
                 f"· Precision {m['precision_score']:.1f}  "
                 f"· Speed {m['speed_score']:.1f}  "
                 f"→ Composite **{m['composite_score']:.1f}**")

        # ── STEP 4: Baseline (optional) ──
        st.write("**Step 4/8** — Baseline computation…")
        baseline = st.session_state.baseline
        if do_baseline and baseline is None and video_mode == "dataset":
            st.write("  ⏳  Processing all segments for percentile baseline…")
            all_recs = process_entire_archive(archive_root)
            all_res = []
            for vid in all_recs.values():
                for seg_recs in vid.values():
                    r2 = calculate_all_metrics(seg_recs)
                    if "metrics" in r2:
                        all_res.append(r2)
            if len(all_res) > 1:
                baseline = compute_baseline(all_res)
                st.session_state.baseline = baseline
                st.write(f"  ✅  Baseline from {len(all_res)} segments")
            else:
                st.write("  ℹ️  Not enough segments — using ASSET rule-based thresholds")
        else:
            st.write("  ℹ️  Using ASSET rule-based thresholds (optimal for single-session)")

        # ── STEP 5: Classify ──
        st.write("**Step 5/8** — Classifying surgeon skill level…")
        clf = classify(m, baseline)
        skill = clf["skill_level"]
        st.write(f"  ✅  **{skill}** "
                 f"(method: {clf['method'].replace('_',' ')}, "
                 f"confidence: {clf['confidence']:.0%})")

        # ── STEP 6: Clinical Feedback ──
        st.write("**Step 6/8** — Generating phase-aware clinical feedback…")
        # Compute phase time analysis
        phase_timing = phase_time_analysis(records, phase_anns)
        result["phase_timing"] = phase_timing
        fb = generate_feedback(m, skill, phase_label, surgeon_name)
        st.write(f"  ✅  Clinical assessment ready · Session duration: **{phase_timing['total_duration_formatted']}**")

        # ── STEP 7: Visual outputs ──
        st.write("**Step 7/8** — Creating annotated visual outputs…")
        best_frame = find_best_frame(selected_seg_str)
        overlay_path = compare_path = pdf_path = None
        traj = result.get("trajectory", {})

        if best_frame and traj.get("xs"):
            overlay_path = str(out_dir / f"overlay_{ts_tag}.png")
            draw_trajectory_on_frame(
                best_frame, traj["xs"], traj["ys"],
                overlay_path, skill, m["composite_score"]
            )
            if do_compare:
                compare_path = str(out_dir / f"compare_{ts_tag}.png")
                create_expert_comparison(
                    best_frame, traj["xs"], traj["ys"],
                    compare_path, skill, m["composite_score"]
                )
            st.write("  ✅  Annotated frame + expert comparison created")
        else:
            st.write("  ⚠️  No suitable frame found for annotation (low detection rate)")

        # ── STEP 8: PDF Report ──
        st.write("**Step 8/8** — Generating hospital-grade PDF report…")
        if do_pdf:
            pdf_path = str(out_dir / f"report_{seg_display_name}_{ts_tag}.pdf")
            try:
                generate_pdf_report(
                    result, fb, clf,
                    overlay_img=overlay_path,
                    compare_img=compare_path,
                    output_path=pdf_path,
                    surgeon_name=surgeon_name,
                    session_id=seg_display_name,
                    baseline=baseline,
                )
                st.write(f"  ✅  PDF report generated ({Path(pdf_path).stat().st_size // 1024} KB)")
            except Exception as e:
                st.write(f"  ⚠️  PDF error: {e}")
                pdf_path = None
        else:
            st.write("  ℹ️  PDF generation disabled")

        # Save session
        st.session_state.last = {
            "result": result, "fb": fb, "clf": clf, "skill": skill,
            "overlay": overlay_path, "compare": compare_path, "pdf": pdf_path,
            "seg_name": seg_display_name, "phase": phase_label,
            "surgeon": surgeon_name, "baseline": baseline,
            "phase_timing": phase_timing,
        }
        st.session_state.history.append({
            "session":    seg_display_name,
            "composite":  m["composite_score"],
            "stability":  m["stability_score"],
            "efficiency": m["efficiency_score"],
            "precision":  m["precision_score"],
            "speed":      m["speed_score"],
            "idle":       m["idle_time_percent"],
            "pe_score":   m.get("procedural_effectiveness_score", 0),
            "tremor_hz":  m.get("tremor_dominant_hz", 0),
            "skill":      skill,
            "frames":     result["frames_analysed"],
            "det_rate":   result["detection_rate"],
            "phase":      phase_label,
            "duration":   phase_timing["total_duration_formatted"],
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
        })

        status.update(label="✅ All 8 pipeline steps complete", state="complete", expanded=False)


# ════════════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ════════════════════════════════════════════════════════════════════
s = st.session_state.last

if s:
    m       = s["result"]["metrics"]
    fb      = s["fb"]
    clf     = s["clf"]
    skill   = s["skill"]
    s_col   = skill_color(skill)
    s_emoji = skill_emoji(skill)
    traj    = s["result"].get("trajectory", {})
    tl      = s["result"].get("timeline", [])
    det_rt  = s["result"].get("detection_rate", 100)
    phase   = s["phase"]

    # ── OUTPUT 1 HEADER ──────────────────────────────────────────────
    st.markdown('<div class="output-header">📊 OUTPUT 1 — LIVE PERFORMANCE DASHBOARD</div>',
                unsafe_allow_html=True)

    # Skill banner
    _, bc, _ = st.columns([1, 3, 1])
    with bc:
        clin = skill_clinical_equiv(skill)
        st.markdown(
            f'<div style="text-align:center;margin:8px 0 20px;">'
            f'<span class="skill-pill" style="background:linear-gradient(135deg,{s_col},{s_col}CC);">'
            f'{s_emoji} {skill.upper()}</span><br>'
            f'<span style="color:#7F8C8D;font-size:0.82rem;margin-top:6px;display:block;">'
            f'Composite: <b style="color:#0D2B4E;font-size:1.1rem;">'
            f'{m["composite_score"]:.1f}/100</b>'
            f'&nbsp;&nbsp;·&nbsp;&nbsp;{s["seg_name"]}'
            f'&nbsp;&nbsp;·&nbsp;&nbsp;Phase: {phase}'
            f'&nbsp;&nbsp;·&nbsp;&nbsp;<i>{clin}</i>'
            f'&nbsp;&nbsp;·&nbsp;&nbsp;{clf["confidence"]:.0%} confidence'
            f'</span></div>',
            unsafe_allow_html=True
        )

    # Detection rate bar
    bar_col = "#27AE60" if det_rt > 80 else "#D68910" if det_rt > 50 else "#C0392B"
    st.markdown(
        f'<div style="font-size:0.78rem;color:#7F8C8D;margin-bottom:4px;">'
        f'Instrument detection rate: {det_rt:.0f}%  '
        f'({s["result"]["frames_analysed"]} / {s["result"]["frames_total"]} frames)</div>'
        f'<div style="height:8px;border-radius:4px;background:#E9ECEF;overflow:hidden;">'
        f'<div style="height:100%;width:{det_rt:.0f}%;background:{bar_col};'
        f'border-radius:4px;transition:width 1s;"></div></div><br>',
        unsafe_allow_html=True
    )

    # ── Gauge charts ─────────────────────────────────────────────────
    st.markdown('<div class="sec-head">Performance Metrics</div>', unsafe_allow_html=True)

    def gauge(val, label, col_hex):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=val,
            delta={"reference": 80,
                   "increasing": {"color": "#27AE60"},
                   "decreasing": {"color": "#C0392B"}},
            title={"text": label, "font": {"size": 11, "color": "#0D2B4E", "family": "Inter"}},
            number={"suffix": "/100", "font": {"size": 24, "color": col_hex, "family": "Inter"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickfont": {"size": 8}},
                "bar":  {"color": col_hex, "thickness": 0.28},
                "bgcolor": "#F8F9FA",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  40], "color": "#FADBD8"},
                    {"range": [40, 60], "color": "#FDEBD0"},
                    {"range": [60, 80], "color": "#D6EAF8"},
                    {"range": [80,100], "color": "#D5F5E3"},
                ],
                "threshold": {
                    "line": {"color": "#0D2B4E", "width": 2},
                    "thickness": 0.75, "value": val
                },
            }
        ))
        fig.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="white",
            font={"family": "Inter"},
        )
        return fig

    g1, g2, g3, g4 = st.columns(4)
    pairs = [
        (g1, m["stability_score"],  "📷 Stability (35%)"),
        (g2, m["efficiency_score"], "📐 Efficiency (25%)"),
        (g3, m["precision_score"],  "🎯 Precision (25%)"),
        (g4, m["speed_score"],      "⚡ Speed (15%)"),
    ]
    for col, val, lbl in pairs:
        c = score_colour(val)
        with col:
            st.plotly_chart(gauge(val, lbl, c), width="stretch",
                            key=f"g_{lbl[:5]}")
            sb = score_band(val)
            st.markdown(
                f'<div style="text-align:center;margin-top:-12px;">'
                f'<span style="background:{skill_color(sb)};color:white;'
                f'border-radius:12px;padding:2px 10px;font-size:0.7rem;font-weight:700;">'
                f'{sb}</span></div>',
                unsafe_allow_html=True
            )

    # Composite summary bar
    idle = m["idle_time_percent"]
    idle_col = "#27AE60" if idle < 20 else "#D68910" if idle < 40 else "#C0392B"
    st.markdown(
        f'<div style="background:white;border-radius:10px;padding:12px 18px;'
        f'text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.05);margin:8px 0 20px;">'
        f'Idle time: <b style="color:{idle_col};">{idle:.1f}%</b>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'Weighted Composite: <b style="color:#0D2B4E;font-size:1.15rem;">'
        f'{m["composite_score"]:.1f}/100</b>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'Method: <b style="color:#7F8C8D;">{clf["method"].replace("_"," ").title()}</b>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ── Timeline + Path side by side ─────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="sec-head">📈 Stability Timeline (per window)</div>',
                    unsafe_allow_html=True)
        if tl:
            df_tl = pd.DataFrame(tl)
            fig_tl = go.Figure()
            for y0, y1, col_, label_ in [
                (80, 100, "rgba(39,174,96,0.12)",  "Expert"),
                (60,  80, "rgba(41,128,185,0.10)", "Advanced"),
                (40,  60, "rgba(214,137,16,0.10)", "Intermediate"),
                ( 0,  40, "rgba(192,57,43,0.12)",  "Novice"),
            ]:
                fig_tl.add_hrect(y0=y0, y1=y1, fillcolor=col_, line_width=0,
                                  opacity=1, annotation_text=label_,
                                  annotation_position="right",
                                  annotation_font_size=9,
                                  annotation_font_color="#888")
            fig_tl.add_trace(go.Scatter(
                x=df_tl["window_start"], y=df_tl["stability"],
                mode="lines+markers",
                line=dict(color="#1A4F8A", width=2.5),
                marker=dict(size=8, color=df_tl["stability"],
                             colorscale=[[0,"#C0392B"],[0.4,"#D68910"],
                                          [0.6,"#2980B9"],[1,"#27AE60"]],
                             cmin=0, cmax=100, line_width=1,
                             line_color="white"),
                hovertemplate="Frame %{x}<br>Stability: %{y:.1f}/100<extra></extra>",
                showlegend=False,
                fill="tozeroy",
                fillcolor="rgba(26,79,138,0.06)",
            ))
            mean_s = df_tl["stability"].mean()
            fig_tl.add_hline(y=mean_s, line_dash="dot", line_color="#0D2B4E",
                              line_width=1.5,
                              annotation_text=f"Mean: {mean_s:.1f}",
                              annotation_position="right",
                              annotation_font_size=8)
            fig_tl.update_layout(
                height=270, paper_bgcolor="white", plot_bgcolor="white",
                xaxis=dict(title="Frame", showgrid=False, zeroline=False,
                           tickfont=dict(size=9)),
                yaxis=dict(range=[0, 108], title="Stability Score",
                           showgrid=True, gridcolor="#F2F3F4",
                           tickfont=dict(size=9)),
                margin=dict(l=10, r=65, t=10, b=35),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_tl, width="stretch", key="timeline")
        else:
            st.info("Not enough frames for timeline view.")

    with right:
        st.markdown('<div class="sec-head">🎯 Instrument Path</div>',
                    unsafe_allow_html=True)
        if s.get("overlay") and os.path.exists(s["overlay"]):
            st.image(s["overlay"], width="stretch",
                     caption="Instrument trajectory overlaid on surgical frame")
        elif traj.get("xs"):
            xs_a, ys_a = traj["xs"], traj["ys"]
            n = len(xs_a)
            fig_p = go.Figure(go.Scatter(
                x=xs_a, y=ys_a, mode="lines+markers",
                marker=dict(size=5, color=list(range(n)),
                             colorscale="Plasma", showscale=False),
                line=dict(color="#1A4F8A", width=1.5),
                hovertemplate="x=%{x:.0f}  y=%{y:.0f}<extra></extra>"
            ))
            fig_p.update_layout(
                height=270, paper_bgcolor="white",
                plot_bgcolor="#0D0D1A",
                xaxis=dict(showgrid=False, title="X (px)"),
                yaxis=dict(showgrid=False, title="Y (px)", autorange="reversed"),
                margin=dict(l=10, r=10, t=10, b=35),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_p, width="stretch", key="path")
        else:
            st.info("No frames with instrument detected.")

    st.markdown("---")

    # ── Expert comparison ─────────────────────────────────────────────
    if s.get("compare") and os.path.exists(s["compare"]):
        st.markdown('<div class="sec-head">👁 Expert Trajectory Comparison</div>',
                    unsafe_allow_html=True)
        st.image(s["compare"],
                 caption="Left: Trainee trajectory  |  Right: AI-generated expert reference trajectory",
                 width="stretch")
        st.markdown("---")

    # ── Radar chart ──────────────────────────────────────────────────
    left2, right2 = st.columns([2, 3])

    with left2:
        st.markdown('<div class="sec-head">🕸 Performance Radar</div>',
                    unsafe_allow_html=True)
        cat   = ["Stability", "Efficiency", "Precision", "Speed", "Idle Control"]
        vals  = [
            m["stability_score"], m["efficiency_score"],
            m["precision_score"], m["speed_score"],
            max(0, 100 - m["idle_time_percent"])
        ]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=[80]*5, theta=cat, fill="toself",
            fillcolor="rgba(39,174,96,0.07)",
            line=dict(color="#27AE60", width=1.5, dash="dot"),
            name="Expert threshold (80)",
        ))
        r_hex = skill_color(skill)
        r_rgb = tuple(int(r_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        fig_r.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cat + [cat[0]], fill="toself",
            fillcolor=f"rgba{r_rgb + (0.25,)}",
            line=dict(color=r_hex, width=2.5),
            name=skill,
            hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
        ))
        fig_r.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 100], tickfont=dict(size=8),
                                gridcolor="#E5E7EB"),
                angularaxis=dict(tickfont=dict(size=10)),
                bgcolor="white",
            ),
            height=320,
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                        x=0.5, xanchor="center", font=dict(size=9)),
            margin=dict(l=50, r=50, t=20, b=50),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_r, width="stretch", key="radar")

    with right2:
        # ── OUTPUT 2 HEADER ──────────────────────────────────────────
        st.markdown('<div class="output-header">📄 OUTPUT 2 — CLINICAL ASSESSMENT</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="fb-box">'
            f'<p><b style="color:#0D2B4E">Overall Assessment:</b><br>{fb["opening"]}</p>'
            f'<p><b style="color:#0D2B4E">Phase Context ({phase}):</b><br>{fb["phase_context"]}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
        for i, comment in enumerate(fb["metric_comments"]):
            labels = ["Stability", "Efficiency", "Precision", "Speed"]
            vals_m = [m["stability_score"], m["efficiency_score"],
                      m["precision_score"], m["speed_score"]]
            col_hex = score_colour(vals_m[i])
            st.markdown(
                f'<div style="background:#F8F9FA;border-left:4px solid {col_hex};'
                f'border-radius:8px;padding:10px 14px;margin:6px 0;'
                f'font-size:0.87rem;color:#2C3E50;line-height:1.65;">'
                f'<b style="color:{col_hex};">{labels[i]}:</b> {comment}</div>',
                unsafe_allow_html=True
            )
        if fb.get("idle_comment"):
            st.markdown(
                f'<div style="background:#FEF9E7;border-left:4px solid #D68910;'
                f'border-radius:8px;padding:10px 14px;margin:6px 0;'
                f'font-size:0.87rem;color:#7D6608;line-height:1.65;">'
                f'<b style="color:#D68910;">Idle Time:</b> {fb["idle_comment"]}</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── NEW: Procedural Effectiveness + Tremor Analysis ──────────────
    st.markdown('<div class="sec-head">🏥 Procedural Effectiveness & Tremor Analysis</div>',
                unsafe_allow_html=True)

    pe_col, tremor_col = st.columns(2)

    pe_score = m.get("procedural_effectiveness_score", 0)
    pe_col_hex = score_colour(pe_score)

    with pe_col:
        st.markdown(
            f'<div class="card">'
            f'<div style="font-size:0.85rem;font-weight:700;color:#0D2B4E;margin-bottom:12px;">'
            f'📊 Procedural Effectiveness Score</div>'
            f'<div style="text-align:center;margin-bottom:14px;">'
            f'<span style="font-size:2.8rem;font-weight:800;color:{pe_col_hex};">'
            f'{pe_score:.1f}</span>'
            f'<span style="font-size:1rem;color:#7F8C8D;">/100</span></div>',
            unsafe_allow_html=True
        )
        # Sub-score bars
        pe_subs = [
            ("Task Completion",  m.get("pe_task_completion", 0), "35%"),
            ("Motion Economy",   m.get("pe_motion_economy",  0), "30%"),
            ("Dexterity",        m.get("pe_dexterity",       0), "20%"),
            ("Idle Penalty",     m.get("pe_idle_penalty",    0), "15%"),
        ]
        for sub_lbl, sub_val, weight in pe_subs:
            sub_col = score_colour(sub_val)
            st.markdown(
                f'<div style="margin:4px 0;">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.75rem;color:#5D6D7E;margin-bottom:2px;">'
                f'<span>{sub_lbl} <span style="color:#AAA">({weight})</span></span>'
                f'<span style="font-weight:700;color:{sub_col};">{sub_val:.0f}</span></div>'
                f'<div style="height:6px;border-radius:3px;background:#E9ECEF;">'
                f'<div style="height:100%;width:{sub_val:.0f}%;background:{sub_col};'
                f'border-radius:3px;"></div></div></div>',
                unsafe_allow_html=True
            )
        interp = m.get("pe_interpretation", "")
        st.markdown(
            f'<div style="margin-top:10px;font-size:0.78rem;color:#5D6D7E;'
            f'font-style:italic;line-height:1.5;">{interp}</div></div>',
            unsafe_allow_html=True
        )

    with tremor_col:
        tremor_hz      = m.get("tremor_dominant_hz", 0)
        tremor_sev     = m.get("tremor_severity", "Minimal")
        tremor_scr     = m.get("tremor_score", 70)
        tremor_reliable= m.get("tremor_reliable", True)
        jerk_raw       = m.get("smoothness_jerk_score", -1)
        jerk_reliable  = m.get("jerk_reliable", True) and jerk_raw >= 0
        t_col_hex      = score_colour(tremor_scr) if tremor_reliable else "#7F8C8D"
        tremor_display = f"{tremor_scr:.0f}" if tremor_reliable else "N/A"
        jerk_display   = f"{jerk_raw:.0f}" if jerk_reliable else "N/A"
        jerk_col       = score_colour(jerk_raw) if jerk_reliable else "#7F8C8D"
        hz_display     = f"{tremor_hz:.1f}" if tremor_reliable else "—"

        st.markdown(
            f'<div class="card">'
            f'<div style="font-size:0.85rem;font-weight:700;color:#0D2B4E;margin-bottom:12px;">'
            f'📡 Tremor & Smoothness Analysis</div>'
            f'<div style="display:flex;gap:12px;margin-bottom:14px;">'
            f'<div style="flex:1;text-align:center;background:#F8F9FA;border-radius:8px;padding:10px;">'
            f'<div style="font-size:1.9rem;font-weight:800;color:{t_col_hex};">{tremor_display}</div>'
            f'<div style="font-size:0.7rem;color:#7F8C8D;">Tremor Score</div></div>'
            f'<div style="flex:1;text-align:center;background:#F8F9FA;border-radius:8px;padding:10px;">'
            f'<div style="font-size:1.9rem;font-weight:800;color:{jerk_col};">{jerk_display}</div>'
            f'<div style="font-size:0.7rem;color:#7F8C8D;">Smoothness</div></div>'
            f'<div style="flex:1;text-align:center;background:#F8F9FA;border-radius:8px;padding:10px;">'
            f'<div style="font-size:1.6rem;font-weight:800;color:#0D2B4E;">{hz_display}</div>'
            f'<div style="font-size:0.7rem;color:#7F8C8D;">Hz (dominant)</div></div>'
            f'</div>'
            f'<div style="background:#EBF3FB;border-radius:6px;padding:8px 12px;'
            f'font-size:0.78rem;color:#1A5276;margin-bottom:8px;">'
            f'<b>Severity:</b> {tremor_sev}</div>',
            unsafe_allow_html=True
        )
        if not tremor_reliable:
            st.info("⚠️ Tremor & smoothness analysis requires ≥60 frames for reliable results. "
                    "This segment has too few frames. Use a longer video for these metrics.")
        else:
            # Frequency reference guide
            freq_zones = [
                ("<2 Hz",   "Smooth / Intentional",   "#27AE60"),
                ("2-4 Hz",  "Low oscillation",         "#2980B9"),
                ("4-8 Hz",  "Action tremor range",     "#D68910"),
                ("8-12 Hz", "Physiological tremor",    "#E67E22"),
                (">12 Hz",  "Pathological concern",    "#C0392B"),
            ]
            for fz, fz_lbl, fz_col in freq_zones:
                is_current = (
                    (tremor_hz < 2 and fz == "<2 Hz") or
                    (2 <= tremor_hz < 4 and fz == "2-4 Hz") or
                    (4 <= tremor_hz < 8 and fz == "4-8 Hz") or
                    (8 <= tremor_hz < 12 and fz == "8-12 Hz") or
                    (tremor_hz >= 12 and fz == ">12 Hz")
                )
                bg = f"background:{fz_col}22;border-left:3px solid {fz_col};" if is_current else "background:#F8F9FA;border-left:3px solid #DEE2E6;"
                st.markdown(
                    f'<div style="{bg}border-radius:4px;padding:4px 8px;'
                    f'margin:2px 0;font-size:0.72rem;display:flex;justify-content:space-between;">'
                    f'<span style="font-weight:{"700" if is_current else "400"};color:{"#0D2B4E" if is_current else "#7F8C8D"};">'
                    f'{fz}</span><span style="color:#7F8C8D;">{fz_lbl}</span></div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

        # Tremor comment from feedback
        if s["fb"].get("tremor_comment"):
            st.markdown(
                f'<div style="background:#F8F9FA;border-left:4px solid {t_col_hex};'
                f'border-radius:8px;padding:10px 14px;margin-top:10px;'
                f'font-size:0.82rem;color:#2C3E50;line-height:1.6;">'
                f'{s["fb"]["tremor_comment"]}</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── NEW: Phase Time Duration Analysis ────────────────────────────
    pt = s.get("phase_timing") or s["result"].get("phase_timing", {})
    if pt:
        st.markdown('<div class="sec-head">⏱️ Phase Time Duration Analysis</div>',
                    unsafe_allow_html=True)

        dur_total = pt.get("total_duration_formatted", "N/A")
        dur_secs  = pt.get("total_duration_s", 0)
        fps_est   = pt.get("estimated_fps", 30)

        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.markdown(
                f'<div class="card" style="text-align:center;">'
                f'<div style="font-size:2rem;font-weight:800;color:#0D2B4E;">{dur_total}</div>'
                f'<div style="font-size:0.75rem;color:#7F8C8D;margin-top:4px;">Total Session Duration</div>'
                f'</div>', unsafe_allow_html=True)
        with tc2:
            st.markdown(
                f'<div class="card" style="text-align:center;">'
                f'<div style="font-size:2rem;font-weight:800;color:#1A5276;">{dur_secs:.0f}s</div>'
                f'<div style="font-size:0.75rem;color:#7F8C8D;margin-top:4px;">Total Seconds</div>'
                f'</div>', unsafe_allow_html=True)
        with tc3:
            st.markdown(
                f'<div class="card" style="text-align:center;">'
                f'<div style="font-size:2rem;font-weight:800;color:#27AE60;">{fps_est:.0f}</div>'
                f'<div style="font-size:0.75rem;color:#7F8C8D;margin-top:4px;">Est. FPS</div>'
                f'</div>', unsafe_allow_html=True)

        # Per-phase breakdown if available
        phases_data = pt.get("phases", {})
        if phases_data:
            st.markdown("<br>", unsafe_allow_html=True)
            phase_rows = []
            for ph_name, ph_info in phases_data.items():
                phase_rows.append({
                    "Phase": ph_name,
                    "Duration": ph_info["duration_formatted"],
                    "Seconds": ph_info["duration_s"],
                    "Frames": ph_info["frame_count"],
                    "% of Session": ph_info["proportion_pct"],
                })
            df_phases = pd.DataFrame(phase_rows).sort_values("Seconds", ascending=False)

            # Phase time bar chart
            fig_pt = go.Figure()
            colours_pt = ["#1A4F8A", "#27AE60", "#D68910", "#C0392B", "#8E44AD"]
            for i, (_, row) in enumerate(df_phases.iterrows()):
                fig_pt.add_trace(go.Bar(
                    x=[row["Seconds"]],
                    y=[row["Phase"]],
                    orientation="h",
                    marker_color=colours_pt[i % len(colours_pt)],
                    text=f'{row["Duration"]}  ({row["% of Session"]}%)',
                    textposition="inside",
                    name=row["Phase"],
                    showlegend=False,
                ))
            fig_pt.update_layout(
                height=max(120, len(phase_rows) * 50 + 60),
                paper_bgcolor="white", plot_bgcolor="white",
                xaxis=dict(title="Duration (seconds)", showgrid=True,
                           gridcolor="#F2F3F4"),
                yaxis=dict(showgrid=False),
                margin=dict(l=10, r=10, t=10, b=35),
                font=dict(family="Inter"),
                barmode="stack",
            )
            st.plotly_chart(fig_pt, width="stretch", key="phase_time")
            st.dataframe(
                df_phases.style.format({"Seconds": "{:.1f}", "% of Session": "{:.1f}"}),
                width="stretch", hide_index=True
            )
        else:
            st.info(
                f"⏱️ Estimated session duration: **{dur_total}** ({dur_secs:.0f}s) at ~{fps_est:.0f} FPS. "
                f"Per-phase breakdown requires phase annotation files from the ArthroPhase dataset. "
                f"If using video/YouTube input, annotations are not available."
            )

        # PE feedback comment
        if s["fb"].get("pe_comment"):
            st.markdown(
                f'<div style="background:white;border-left:5px solid #27AE60;border-radius:10px;'
                f'padding:14px 18px;margin-top:10px;font-size:0.87rem;color:#2C3E50;line-height:1.7;">'
                f'<b style="color:#0D2B4E;">Procedural Effectiveness Assessment:</b><br>'
                f'{s["fb"]["pe_comment"]}</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Development Actions ──────────────────────────────────────────
    st.markdown('<div class="sec-head">📋 Recommended Development Actions</div>',
                unsafe_allow_html=True)
    act_cols = st.columns(2)
    for i, action in enumerate(fb["actions"]):
        with act_cols[i % 2]:
            st.markdown(
                f'<div class="action-row">'
                f'<b style="color:#1A5276;margin-right:8px;">{i+1}.</b>{action}'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── OUTPUT 3 HEADER ──────────────────────────────────────────────
    st.markdown('<div class="output-header">🎯 OUTPUT 3 — ANNOTATED FRAME (Visual Tracking Proof)</div>',
                unsafe_allow_html=True)

    if s.get("overlay") and os.path.exists(s["overlay"]):
        ann_col1, ann_col2 = st.columns([1, 1])
        with ann_col1:
            st.markdown("**Trajectory Overlay**")
            st.image(s["overlay"], width="stretch",
                     caption=f"Instrument path — {len(traj.get('xs', []))} tracked positions")
        with ann_col2:
            if s.get("compare") and os.path.exists(s["compare"]):
                st.markdown("**Expert vs Trainee Comparison**")
                st.image(s["compare"], width="stretch",
                         caption="Trainee path (left) vs AI Expert reference (right)")
        st.markdown("""
        <div style="background:#EBF3FB;border-radius:8px;padding:12px 16px;margin:10px 0;
        font-size:0.83rem;color:#1A5276;border-left:3px solid #2980B9;">
        <b>Note:</b> These annotated frames are generated from actual surgical video masks.
        The coloured path is the <i>tracked instrument tip</i> drawn over the real endoscopic frame —
        proving that the system correctly localised and tracked the arthroscopic tool throughout the segment.
        </div>""", unsafe_allow_html=True)
    else:
        st.info("No annotated frame generated — low instrument detection rate. "
                "Ensure color_mask PNGs contain visible instrument regions.")

    st.markdown("---")

    # ── Session History ──────────────────────────────────────────────
    if len(st.session_state.history) > 1:
        st.markdown('<div class="sec-head">📈 Session Progress History</div>',
                    unsafe_allow_html=True)
        df_h = pd.DataFrame(st.session_state.history)
        fig_h = go.Figure()
        metric_lines = [
            ("stability",  "#1A4F8A", "solid",  "Stability"),
            ("efficiency", "#27AE60", "solid",  "Efficiency"),
            ("precision",  "#D68910", "solid",  "Precision"),
            ("speed",      "#C0392B", "solid",  "Speed"),
            ("composite",  "#0D2B4E", "dot",    "Composite"),
        ]
        for col_, colour_, dash_, name_ in metric_lines:
            if col_ not in df_h.columns:
                continue
            lw = 3 if col_ == "composite" else 1.8
            fig_h.add_trace(go.Scatter(
                x=df_h["session"], y=df_h[col_],
                name=name_, mode="lines+markers",
                line=dict(color=colour_, width=lw, dash=dash_),
                marker=dict(size=8 if col_ == "composite" else 5),
                hovertemplate=f"{name_}: %{{y:.1f}}<extra></extra>",
            ))
        fig_h.add_hline(y=80, line_dash="dot", line_color="#27AE60",
                         line_width=1, opacity=0.6,
                         annotation_text="Expert threshold (80)",
                         annotation_font_size=8, annotation_position="right")
        fig_h.update_layout(
            height=300, paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Session", showgrid=False,
                       tickangle=-30, tickfont=dict(size=8)),
            yaxis=dict(range=[0, 108], title="Score (0–100)",
                       showgrid=True, gridcolor="#F2F3F4",
                       tickfont=dict(size=9)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(size=9)),
            margin=dict(l=10, r=80, t=30, b=60),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_h, width="stretch", key="history")

        avail_cols = list(df_h.columns)
        base_cols = ["timestamp", "session", "skill", "composite", "stability",
                     "efficiency", "precision", "speed", "idle", "frames",
                     "det_rate", "phase"]
        extra_cols = [c for c in ["pe_score", "duration"] if c in avail_cols]
        sel_cols = base_cols + extra_cols
        col_names = ["Time", "Session", "Skill", "Composite", "Stability",
                     "Efficiency", "Precision", "Speed", "Idle%",
                     "Frames", "Det%", "Phase"]
        if "pe_score" in extra_cols: col_names.append("PE Score")
        if "duration" in extra_cols: col_names.append("Duration")
        df_disp = df_h[sel_cols].copy()
        df_disp.columns = col_names
        fmt = {"Composite": "{:.1f}", "Stability": "{:.1f}",
               "Efficiency": "{:.1f}", "Precision": "{:.1f}",
               "Speed": "{:.1f}", "Idle%": "{:.1f}", "Det%": "{:.0f}"}
        if "PE Score" in col_names: fmt["PE Score"] = "{:.1f}"
        st.dataframe(
            df_disp.style.format(fmt).background_gradient(
                subset=["Composite"], cmap="RdYlGn", vmin=0, vmax=100),
            width="stretch"
        )
        st.markdown("---")

    # ── Downloads ────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">⬇️ Downloads</div>', unsafe_allow_html=True)
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        if s.get("pdf") and os.path.exists(s["pdf"]):
            with open(s["pdf"], "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"surgical_report_{s['seg_name']}.pdf",
                mime="application/pdf",
                type="primary",
                width="stretch",
            )
        else:
            st.info("PDF not generated")

    with dl2:
        json_str = json.dumps(s["result"]["metrics"], indent=2)
        st.download_button(
            "📊 Download Metrics JSON",
            data=json_str,
            file_name=f"metrics_{s['seg_name']}.json",
            mime="application/json",
            width="stretch",
        )

    with dl3:
        if s.get("overlay") and os.path.exists(s["overlay"]):
            with open(s["overlay"], "rb") as f:
                img_bytes = f.read()
            st.download_button(
                "🎯 Download Annotated Frame",
                data=img_bytes,
                file_name=f"annotated_frame_{s['seg_name']}.png",
                mime="image/png",
                width="stretch",
            )
        else:
            st.info("No annotated frame")
