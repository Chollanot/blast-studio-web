"""
Blast-YOLO Studio - multi-mode Streamlit app for embryologists.
RESEARCH USE ONLY. Not a regulated medical device.

Modes (sidebar):
  Home          - about + research disclaimer
  Quick Predict - upload one image, get grade + overlay + PDF
  Learn         - tutorial: guess first, then see AI + explanation
  Proficiency   - timed quiz, scored against ground truth + AI
  Cohort        - upload multiple embryos, ranked transfer list
  Batch         - process a ZIP, get CSV + summary statistics
  XAI Explorer  - CAM / segmentation visualisation per image
"""
from __future__ import annotations
import io, os, sys, time, zipfile, json, random
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from inference import BlastocystGrader
from report_generator import build_pdf_report, build_pt_certificate
import learning_resources as LR
import pipeline as P


# ============================================================

# __WEIGHT_DOWNLOAD_BLOCK__
import os as _os, requests as _req
_WEIGHT_URL = 'https://huggingface.co/Chollanot/blast-yolo/resolve/main/best.pt'
_WEIGHT_PATH = 'best.pt'
if not _os.path.exists(_WEIGHT_PATH):
    print('Downloading weights from HF...')
    with open(_WEIGHT_PATH, 'wb') as _f:
        _f.write(_req.get(_WEIGHT_URL, allow_redirects=True).content)
    print(f'Downloaded {_os.path.getsize(_WEIGHT_PATH)/1e6:.1f} MB')


st.set_page_config(page_title='Blast-YOLO Studio',
                   page_icon='DNA', layout='wide',
                   initial_sidebar_state='expanded')

# Global styles
st.markdown("""
<style>
.research-banner {
    background:#fff3cd; border:1px solid #ffeeba; color:#856404;
    padding:10px 14px; border-radius:8px; margin-bottom:14px;
    font-size:13px;
}
.metric-card {background:#f5f7fb; padding:16px; border-radius:12px;
              border:1px solid #e1e6ef; margin-bottom:12px;}
.grade-big {font-size:54px; font-weight:700; color:#1f3a8a; margin:0;}
.grade-sub {color:#475569; margin:0;}
.priority-1 {background:#dcfce7; padding:4px 10px; border-radius:6px;
             color:#166534; font-weight:600;}
.priority-2 {background:#fef3c7; padding:4px 10px; border-radius:6px;
             color:#92400e; font-weight:600;}
.priority-3 {background:#fee2e2; padding:4px 10px; border-radius:6px;
             color:#991b1b; font-weight:600;}
</style>""", unsafe_allow_html=True)

# Persistent research-only banner
st.markdown(
    "<div class='research-banner'>"
    "<b>RESEARCH USE ONLY.</b> This application is an investigational "
    "decision-support tool. It is not a regulated medical device and "
    "must not be used as the sole basis for any clinical decision. "
    "All inference runs locally; no embryo image leaves this device."
    "</div>", unsafe_allow_html=True)

# ----- Sidebar: model + navigation -----
st.sidebar.title('Blast-YOLO Studio')
st.sidebar.caption('Research tool for embryologists')
WEIGHTS = st.sidebar.text_input('Model weights', str(ROOT / 'best.pt'))
DEVICE  = st.sidebar.selectbox(
    'Device', ['cuda','cpu'] if torch.cuda.is_available() else ['cpu'])

@st.cache_resource(show_spinner=False)
def load_grader(w, d): return BlastocystGrader(w, device=d)

with st.spinner('Loading model...'):
    try:
        grader = load_grader(WEIGHTS, DEVICE)
        st.sidebar.success(f'Loaded ({grader.model_name})')
    except Exception as e:
        st.sidebar.error(f'Failed: {e}'); st.stop()

mode = st.sidebar.radio('Mode', [
    'Home',
    'Quick Predict',
    'Learn',
    'Proficiency Test',
    'Cohort - Transfer ranking',
    'Batch - Research',
    'XAI Explorer',
])

# Persist learning + PT state across reruns
if 'learn_history' not in st.session_state:
    st.session_state.learn_history = []
if 'pt_state' not in st.session_state:
    st.session_state.pt_state = {}


# ============================================================
def _grade_with_explanation(img, embryo_id=''):
    """Return result dict augmented with a learning-friendly explanation."""
    r = grader.predict(img)
    r['explanation'] = LR.grade_explanation(r['grade'])
    r['embryo_id']   = embryo_id
    r['rank']        = LR.transfer_rank(r['grade'])
    return r


def _priority_tag(grade):
    rank = LR.transfer_rank(grade)
    if rank < 4:  return f"<span class='priority-1'>Top priority</span>"
    if rank < 12: return f"<span class='priority-2'>Good / Fair</span>"
    if rank < 23: return f"<span class='priority-3'>Low / Caution</span>"
    return        f"<span class='priority-3'>Needs review</span>"


# ============================================================
# MODE: HOME
# ============================================================
if mode == 'Home':
    st.title('Blast-YOLO Studio')
    st.markdown("""
**For embryologists**, at every career stage.

This is a research-use app that wraps the Blast-YOLO multi-task model for
24-class blastocyst grading. It supports four workflows:

| Mode | Best for | Purpose |
|---|---|---|
| **Quick Predict** | Any user | Single-image grading with overlay + PDF report |
| **Learn** | Junior embryologists, students | Guess first, then see the AI prediction and a plain-English explanation of every grade |
| **Proficiency Test** | All staff (annual PT / refresher) | Timed quiz, scored against ground truth and the AI; downloadable certificate |
| **Cohort - Transfer ranking** | Senior embryologists, physicians | Upload a whole cohort, get a ranked transfer list with clinical-action tags (P1, LP, REV) |
| **Batch - Research** | Researchers | Process ZIP of images, get CSV + summary statistics |
| **XAI Explorer** | Anyone | See what regions the model attends to (Grad-CAM, segmentation) |

### Important reminders
- **Research use only** - not a regulated medical device.
- **No PHI leaves this device.** All inference runs locally on the laboratory computer.
- The trained embryologist remains responsible for the final clinical grade.
- The Review (REV) class deliberately has no fixed morphology - the model abstains there by design.

### Quick start
1. Make sure `best.pt` is next to `app.py` (or set the path in the sidebar).
2. Pick a mode from the sidebar.
3. Read the brief instructions at the top of each mode page.
""")

# ============================================================
# MODE: QUICK PREDICT
# ============================================================
elif mode == 'Quick Predict':
    st.title('Quick Predict')
    st.caption('Upload one blastocyst image -> grade, segmentation overlay, '
               'top-3 alternatives, PDF report.')

    up  = st.file_uploader('Blastocyst image', type=['jpg','jpeg','png'])
    pid = st.text_input('Patient / cycle ID (optional)')
    eid = st.text_input('Embryo ID (optional)')

    if up is not None:
        img = Image.open(up).convert('RGB')
        c1, c2 = st.columns([1, 1])
        with c1: st.image(img, caption='Input', use_column_width=True)
        with st.spinner('Inference...'):
            t0 = time.time(); r = _grade_with_explanation(img, eid); t = time.time()-t0
        with c2:
            st.markdown(f"""<div class='metric-card'>
                <p class='grade-sub'>Predicted Gardner grade</p>
                <p class='grade-big'>{r['grade']}</p>
                <p class='grade-sub'>Confidence <b>{r['confidence']*100:.1f}%</b>
                . Clinical group <b>{r['clinical_group']}</b>
                . {t*1000:.0f} ms</p>
                <p class='grade-sub'>{_priority_tag(r['grade'])}</p>
                </div>""", unsafe_allow_html=True)
            st.markdown('**What this grade means**')
            st.markdown(r['explanation'])
            st.markdown('**Top-3 alternatives**')
            st.dataframe(pd.DataFrame(r['top3'], columns=['grade','probability'])
                         .round(4), use_container_width=True, hide_index=True)
            if r.get('overlay') is not None:
                st.image(r['overlay'],
                         caption='Segmentation overlay: ICM (red) / TE (green) / Cavity (blue)',
                         use_column_width=True)
        st.subheader('All-class confidence')
        st.bar_chart(pd.Series(r['probs'], index=grader.class_names))

        if st.button('Generate PDF report'):
            pdf = build_pdf_report(img, r, patient=pid, embryo=eid,
                                   model=grader.model_name)
            st.download_button('Download PDF', pdf,
                               file_name=f'blast_{eid or "embryo"}.pdf',
                               mime='application/pdf')

# ============================================================
# MODE: LEARN
# ============================================================
elif mode == 'Learn':
    st.title('Learn mode - guess first, then see the AI')
    st.caption("For junior embryologists upskilling and seniors refreshing. "
               "Upload an image, guess the grade, then reveal the AI prediction "
               "with a plain-English explanation.")

    up = st.file_uploader('Blastocyst image', type=['jpg','jpeg','png'],
                          key='learn_up')
    if up is not None:
        img = Image.open(up).convert('RGB')
        st.image(img, width=400)
        guess = st.selectbox('Your grade guess', P.CLASS_NAMES, key='learn_guess')
        if st.button('Reveal AI prediction'):
            r = _grade_with_explanation(img)
            agreed = (guess == r['grade'])
            cA, cB = st.columns(2)
            with cA:
                st.markdown(f"### Your guess: **{guess}**")
                st.markdown(LR.grade_explanation(guess))
            with cB:
                st.markdown(f"### AI prediction: **{r['grade']}** "
                            f"({r['confidence']*100:.1f}%)")
                st.markdown(r['explanation'])
                if r.get('overlay') is not None:
                    st.image(r['overlay'],
                             caption='ICM / TE / Cavity overlay',
                             use_column_width=True)
            if agreed:
                st.success('You agree with the AI on this image.')
            else:
                st.warning('You disagree with the AI. Consider both '
                           'interpretations - manual Gardner grading has '
                           'substantial inter-observer variability and the '
                           'AI is not infallible. Use the overlay to study '
                           'the ICM and trophectoderm regions.')
            st.markdown('**Top-3 AI alternatives**')
            st.dataframe(pd.DataFrame(r['top3'], columns=['grade','p']).round(4),
                         use_container_width=True, hide_index=True)
            st.session_state.learn_history.append({
                'time': datetime.now().isoformat(timespec='seconds'),
                'guess': guess, 'ai_grade': r['grade'],
                'ai_conf': r['confidence'], 'agreed': agreed})

    if st.session_state.learn_history:
        with st.expander('Your learning session history'):
            df = pd.DataFrame(st.session_state.learn_history)
            n  = len(df); agree = int(df['agreed'].sum())
            st.write(f"Total images: {n} . Agreed with AI: {agree} "
                     f"({agree/n*100:.1f}%)")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button('Export session CSV',
                df.to_csv(index=False).encode(),
                file_name='learn_session.csv', mime='text/csv')

# ============================================================
# MODE: PROFICIENCY TEST
# ============================================================
elif mode == 'Proficiency Test':
    st.title('Proficiency Test (PT)')
    st.caption("Timed quiz over a set of labelled blastocyst images. "
               "Scored against ground truth and against the AI prediction.")

    pt = st.session_state.pt_state
    if not pt:
        st.markdown("""**To start a PT session:** upload a ZIP archive whose
filenames encode the ground-truth grade (e.g. `123 4AB.jpg`,
`emb_3BB.jpg`). The grade is parsed from the filename. We recommend at
least 10 images for a meaningful score.""")
        examinee = st.text_input("Examinee name (printed on the certificate)")
        n_q = st.slider('Number of questions', 5, 50, 10)
        timed = st.checkbox('Time-limit each question (60 s)', value=True)
        zf = st.file_uploader('ZIP of labelled blastocyst images', type=['zip'])
        if zf is not None and examinee and st.button('Start PT'):
            import re
            pat = re.compile(r'\b(\d?[A-Z]{2,3}|LP|P1|REV)\b')
            imgs = []
            with zipfile.ZipFile(zf) as z:
                for n in z.namelist():
                    if not n.lower().endswith(('jpg','jpeg','png')): continue
                    m = pat.search(os.path.basename(n).upper())
                    if not m: continue
                    g = m.group(1)
                    if g in P.CLASS_NAMES:
                        imgs.append((n, z.read(n), g))
            if len(imgs) < n_q:
                st.error(f'Only {len(imgs)} valid labelled images found.')
            else:
                random.shuffle(imgs)
                st.session_state.pt_state = {
                    'examinee': examinee, 'imgs': imgs[:n_q], 'idx': 0,
                    'answers': [], 'started': datetime.now(),
                    'timed': timed,
                }
                st.rerun()

    else:
        i = pt['idx']
        if i < len(pt['imgs']):
            fname, data, gt = pt['imgs'][i]
            img = Image.open(io.BytesIO(data)).convert('RGB')
            st.progress(i / len(pt['imgs']))
            st.markdown(f"**Question {i+1} / {len(pt['imgs'])}**")
            st.image(img, width=380, caption=os.path.basename(fname))
            ans = st.selectbox('Your grade', P.CLASS_NAMES,
                               key=f'pt_a_{i}')
            if st.button('Submit answer', key=f'pt_s_{i}'):
                r = grader.predict(img)
                pt['answers'].append({
                    'q': i+1, 'file': os.path.basename(fname),
                    'truth': gt, 'examinee': ans,
                    'ai': r['grade'], 'ai_conf': r['confidence'],
                    'examinee_correct': ans == gt,
                    'ai_correct': r['grade'] == gt,
                })
                pt['idx'] = i + 1
                st.rerun()
        else:
            # PT finished
            df = pd.DataFrame(pt['answers'])
            total = len(df)
            ex_score = int(df['examinee_correct'].sum())
            ai_score = int(df['ai_correct'].sum())
            elapsed = datetime.now() - pt['started']

            st.success(f"PT session complete - {pt['examinee']}")
            c1, c2, c3 = st.columns(3)
            c1.metric('Your score', f'{ex_score} / {total}',
                      f'{ex_score/total*100:.1f}%')
            c2.metric('AI score',   f'{ai_score} / {total}',
                      f'{ai_score/total*100:.1f}%')
            c3.metric('Time taken', f"{elapsed.seconds//60} min "
                                    f"{elapsed.seconds%60} s")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button('Export PT results CSV',
                df.to_csv(index=False).encode(),
                file_name=f'PT_{pt["examinee"]}_{datetime.now():%Y%m%d}.csv',
                mime='text/csv')

            cert = build_pt_certificate(pt['examinee'], total, ex_score, ai_score,
                                        elapsed, model=grader.model_name)
            st.download_button('Download PT certificate (PDF)', cert,
                file_name=f'PT_certificate_{pt["examinee"]}.pdf',
                mime='application/pdf')
            if st.button('Start a new PT session'):
                st.session_state.pt_state = {}
                st.rerun()

# ============================================================
# MODE: COHORT - Transfer ranking
# ============================================================
elif mode == 'Cohort - Transfer ranking':
    st.title('Cohort - Transfer ranking')
    st.caption("Upload all blastocysts from one patient/cycle. The AI grades "
               "every embryo and ranks the cohort by transfer priority. "
               "**Always reviewed by the embryologist / physician** before "
               "any clinical decision.")

    patient = st.text_input('Patient / cycle ID')
    zf = st.file_uploader('ZIP of cohort blastocyst images', type=['zip'])
    if zf is not None and patient:
        rows = []
        with zipfile.ZipFile(zf) as z:
            names = [n for n in z.namelist()
                     if n.lower().endswith(('jpg','jpeg','png'))]
            prog = st.progress(0.0)
            for i, n in enumerate(names):
                im = Image.open(io.BytesIO(z.read(n))).convert('RGB')
                r  = _grade_with_explanation(im, embryo_id=os.path.basename(n))
                rows.append({
                    'embryo': os.path.basename(n),
                    'grade':  r['grade'],
                    'confidence':     round(r['confidence'], 4),
                    'clinical_group': r['clinical_group'],
                    'action':  r['action'] or '-',
                    'rank':    r['rank'],
                })
                prog.progress((i+1)/len(names))
        df = pd.DataFrame(rows).sort_values('rank').reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = 'transfer_order'

        st.subheader(f"Patient {patient} - {len(df)} embryos ranked")
        st.dataframe(df.drop(columns=['rank']), use_container_width=True)

        # Summary by clinical group
        st.subheader('Cohort summary')
        summary = df['clinical_group'].value_counts().reindex(
            ['top priority','good quality','fair quality',
             'low potential','needs review'], fill_value=0)
        st.bar_chart(summary)

        st.info("**Recommended next step:** the embryologist reviews the top "
                "1-3 candidates against the segmentation overlays (Quick "
                "Predict mode), and the physician confirms the final "
                "transfer decision in the context of the patient's clinical "
                "history.")
        st.download_button('Download cohort CSV',
            df.to_csv().encode('utf-8'),
            file_name=f'cohort_{patient}.csv', mime='text/csv')

# ============================================================
# MODE: BATCH - Research
# ============================================================
elif mode == 'Batch - Research':
    st.title('Batch - Research')
    st.caption("Process a large ZIP of images and export grade + confidence "
               "for downstream analysis. Useful for retrospective audits "
               "and AI-vs-embryologist concordance studies.")
    zf = st.file_uploader('ZIP of blastocyst images', type=['zip'])
    if zf is not None:
        rows = []
        with zipfile.ZipFile(zf) as z:
            names = [n for n in z.namelist()
                     if n.lower().endswith(('jpg','jpeg','png'))]
            st.info(f'{len(names)} images')
            prog = st.progress(0.0)
            for i, n in enumerate(names):
                im = Image.open(io.BytesIO(z.read(n))).convert('RGB')
                r  = grader.predict(im)
                rows.append({
                    'filename': os.path.basename(n),
                    'grade': r['grade'],
                    'confidence': round(r['confidence'], 4),
                    'clinical_group': r['clinical_group'],
                })
                prog.progress((i+1)/len(names))
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.subheader('Grade distribution in this batch')
        st.bar_chart(df['grade'].value_counts())
        st.download_button('Download CSV',
            df.to_csv(index=False).encode('utf-8'),
            file_name='blast_batch.csv', mime='text/csv')

# ============================================================
# MODE: XAI EXPLORER
# ============================================================
else:   # XAI Explorer
    st.title('XAI Explorer')
    st.caption("Visualise what the model attended to. Uses Grad-CAM and "
               "the Blast-YOLO segmentation head.")
    up = st.file_uploader('Image', type=['jpg','jpeg','png'], key='xai_up')
    if up is not None:
        img = Image.open(up).convert('RGB')
        r   = _grade_with_explanation(img)
        c1, c2 = st.columns(2)
        with c1: st.image(img, caption='Original', use_column_width=True)
        with c2:
            if r.get('overlay') is not None:
                st.image(r['overlay'],
                         caption='ICM (red) / TE (green) / Cavity (blue)',
                         use_column_width=True)
        st.markdown(f"**AI grade:** {r['grade']}  "
                    f"({r['confidence']*100:.1f}%) - {r['action']}")
        st.markdown(r['explanation'])
        st.info("To explore other attribution methods (Grad-CAM++, Score-CAM, "
                "Eigen-CAM, Integrated Gradients), use the standalone "
                "`xai.py` module in the research notebook.")

# ===== Footer =====
st.divider()
st.caption("Blast-YOLO Studio - Research use only. Local inference only.")
