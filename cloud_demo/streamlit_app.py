################################## Audio Analyzer — Web Demo ##################################
#
# PUBLIC WEB DEMO of the BirdNET UK & Ireland Audio Analyzer.
#
# This is a deliberately reduced build for Streamlit Community Cloud, intended for
# demonstration to funders and partners. It runs BirdNET only.
#
# The full desktop version (app2.py, in the repository root) additionally provides:
#   - Perch classification (~10,000 species, global model)
#   - Perch embeddings + acoustic similarity search ("Find Similar")
#   - Unrestricted file sizes and recording lengths
#   - A persistent results database
#
# Perch is excluded here on purpose. It pulls in PyTorch (via opensoundscape)
# alongside TensorFlow, and loading both native runtimes into a single process
# segfaults on Community Cloud's environment. Keeping this build BirdNET-only
# also keeps it inside the platform's memory and disk limits.
#
# Developed by Mike Shewring / RSPB Centre for Conservation Science.
# mike.shewring@rspb.org.uk
#
# Run locally from the REPOSITORY ROOT (not from this directory), so that paths
# resolve the same way they do on Community Cloud:
#   streamlit run cloud_demo/streamlit_app.py

import os

# Must be set before any import that loads a native runtime (librosa/numba,
# tflite). Constrains thread pools and prevents duplicate OpenMP loading.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import io
import sqlite3
import tempfile
import traceback
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                      # headless backend — no display on the server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import librosa
import librosa.display
import plotly.express as px
import soundfile as sf


# --- demo limits -------------------------------------------------------------
# Community Cloud containers are memory-limited and shared. These caps keep the
# demo responsive and prevent a single large upload from killing the container.

MAX_FILE_MB = 25            # per-file upload cap (also enforced in .streamlit/config.toml)
MAX_ANALYSIS_SEC = 300      # only the first N seconds of any recording are analysed
MAX_BATCH_FILES = 10        # cap on files per batch run
PREVIEW_SR = 16000          # sample rate for spectrogram rendering only

# Ephemeral: Community Cloud wipes the filesystem on reboot and redeploy.
DB_PATH = Path(tempfile.gettempdir()) / 'birdnet_demo.db'

STATUS_UNREVIEWED = 'Unreviewed'
STATUS_ACCEPTED = 'Accepted'
STATUS_REJECTED = 'Rejected'
STATUS_UNSURE = 'Unsure'

import streamlit as st


# --- model loading -----------------------------------------------------------
# The BirdNET import is deliberately deferred into this cached function rather
# than sitting at module scope. If the TFLite backend fails to load, the failure
# surfaces as a caught exception in the UI instead of taking down the whole app
# during startup.

@st.cache_resource(show_spinner='Loading the BirdNET model (first run only)…')
def load_analyzer():
    from birdnetlib.analyzer import Analyzer
    return Analyzer()


def get_analyzer():
    """Return the analyzer, or None plus an error message if it won't load."""
    try:
        return load_analyzer(), None
    except Exception:
        return None, traceback.format_exc(limit=3)


# --- database helpers --------------------------------------------------------

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id        TEXT,
            file_name       TEXT,
            start_time      REAL,
            end_time        REAL,
            common_name     TEXT,
            scientific_name TEXT,
            confidence      REAL,
            lat             REAL,
            lon             REAL,
            rec_date        TEXT,
            source          TEXT DEFAULT 'birdnet',
            verified        TEXT DEFAULT 'Unreviewed',
            notes           TEXT DEFAULT ''
        )
    """)
    con.commit()
    con.close()


def save_detections_db(batch_id, df):
    con = sqlite3.connect(DB_PATH)
    d = df.copy()
    d['batch_id'] = batch_id
    d.to_sql('detections', con, if_exists='append', index=False)
    con.close()


def load_detections_db(batch_id=None):
    con = sqlite3.connect(DB_PATH)
    if batch_id and batch_id != 'All':
        df = pd.read_sql('SELECT * FROM detections WHERE batch_id=?', con, params=(batch_id,))
    else:
        df = pd.read_sql('SELECT * FROM detections', con)
    con.close()
    return df


def update_verified_db(det_id, status, notes=''):
    con = sqlite3.connect(DB_PATH)
    con.execute('UPDATE detections SET verified=?, notes=? WHERE id=?',
                (status, notes, det_id))
    con.commit()
    con.close()


def get_batch_ids():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute('SELECT DISTINCT batch_id FROM detections ORDER BY batch_id DESC')
    ids = [r[0] for r in cur.fetchall()]
    con.close()
    return ids


# --- spectrogram helpers -----------------------------------------------------

def _mel_db(y, sr):
    return librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)


def plot_annotated_sonogram(y, sr, detections, filename, t_start_sec=0, t_end_sec=60):
    y_slice = y[int(t_start_sec * sr) : min(int(t_end_sec * sr), len(y))]

    fig, ax = plt.subplots(figsize=(14, 4))
    librosa.display.specshow(_mel_db(y_slice, sr), x_axis='time', y_axis='mel',
                             sr=sr, ax=ax, cmap='magma')

    window_dets = [d for d in detections
                   if d['end_time'] >= t_start_sec and d['start_time'] <= t_end_sec]
    species_list = list(set(d['common_name'] for d in window_dets))
    cmap = {sp: plt.cm.tab20(i / max(len(species_list), 1))
            for i, sp in enumerate(species_list)}

    for d in window_dets:
        rel_s = max(d['start_time'] - t_start_sec, 0)
        rel_e = min(d['end_time'] - t_start_sec, t_end_sec - t_start_sec)
        colour = cmap[d['common_name']]
        ax.axvspan(rel_s, rel_e, alpha=0.35, color=colour)
        ax.text(rel_s + (rel_e - rel_s) / 2, ax.get_ylim()[1] * 0.92,
                f"{d['common_name']}\n{d['confidence']:.0%}",
                ha='center', va='top', fontsize=7, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=colour, alpha=0.7))

    if species_list:
        patches = [mpatches.Patch(color=cmap[sp], label=sp) for sp in sorted(species_list)]
        ax.legend(handles=patches, loc='upper right', fontsize=7,
                  framealpha=0.6, ncol=max(1, len(species_list) // 6))

    ax.set_title(f'{filename}  [{t_start_sec:.0f}s – {t_end_sec:.0f}s]', fontsize=10, pad=8)
    ax.set_xlabel('Time in window (s)')
    ax.set_ylabel('Frequency (mel)')
    fig.tight_layout()
    return fig


def plot_detection_sonogram(y, sr, det, pad=1.0):
    total_dur = librosa.get_duration(y=y, sr=sr)
    t_start = max(det['start_time'] - pad, 0)
    t_end = min(det['end_time'] + pad, total_dur)
    y_snip = y[int(t_start * sr) : int(t_end * sr)]

    fig, ax = plt.subplots(figsize=(8, 2))
    librosa.display.specshow(_mel_db(y_snip, sr), x_axis='time', y_axis='mel',
                             sr=sr, ax=ax, cmap='magma')
    ax.axvspan(pad, pad + (det['end_time'] - det['start_time']), alpha=0.3, color='white')
    ax.set_title(f"{det['common_name']}  {det['start_time']:.1f}s–{det['end_time']:.1f}s",
                 fontsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.tight_layout()
    return fig, y_snip


# --- analysis ----------------------------------------------------------------

def run_birdnet(file_path, lat, lon, conf, use_date, rec_date, rec_time):
    from birdnetlib import Recording

    analyzer, err = get_analyzer()
    if analyzer is None:
        raise RuntimeError(f'BirdNET model failed to load:\n{err}')

    kwargs = dict(lat=lat, lon=lon, min_conf=conf)
    if use_date and rec_date:
        kwargs['date'] = datetime.combine(rec_date, rec_time or datetime.now().time())

    recording = Recording(analyzer, str(file_path), **kwargs)
    recording.analyze()

    # Demo cap: discard anything past the analysis window.
    return [d for d in recording.detections if d['start_time'] < MAX_ANALYSIS_SEC]


def write_temp_upload(uploaded_file):
    """Persist an uploaded file to a temp path and return it."""
    suffix = Path(uploaded_file.name).suffix or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


# --- validation state helpers ------------------------------------------------

def vstate():
    if 'validation' not in st.session_state:
        st.session_state['validation'] = {}
    return st.session_state['validation']


def get_vstatus(key):
    return vstate().get(key, STATUS_UNREVIEWED)


def set_vstatus(key, status):
    vstate()[key] = status


# --- app startup -------------------------------------------------------------

st.set_page_config(page_title='Audio Analyzer — Web Demo', layout='wide')
init_db()

st.title('🦆 Audio Analyzer 🦉')
st.caption('BirdNET acoustic analysis for UK & Ireland — public web demonstration')

with st.expander('ℹ️ About this demo', expanded=True):
    st.markdown(f"""
This is a **public demonstration build**. It runs the BirdNET classifier over uploaded
audio, overlays detections on a scrolling spectrogram, and provides a review workflow
for accepting or rejecting each detection.

**Demo limits**

- Uploads capped at **{MAX_FILE_MB} MB** per file
- Only the first **{MAX_ANALYSIS_SEC // 60} minutes** of each recording are analysed
- Up to **{MAX_BATCH_FILES} files** per batch
- Results are **not saved** — everything is cleared when the app restarts

**The full desktop version** additionally runs Google's Perch model (~10,000 species),
generates acoustic embeddings for similarity search across unlabelled audio, and
processes full ARU deployments with a persistent results database.
Contact Mike Shewring for access.
""")

st.sidebar.header('Analysis Settings')

st.sidebar.subheader('Location')
lat = st.sidebar.number_input('Latitude', value=54.0, format='%.4f')
lon = st.sidebar.number_input('Longitude', value=-4.5, format='%.4f')
st.sidebar.caption('Default: central UK/Ireland.')

st.sidebar.subheader('Recording Date (optional)')
use_date = st.sidebar.checkbox('Apply seasonal date filter', value=False)
if use_date:
    rec_date = st.sidebar.date_input('Recording date', value=date.today())
    rec_time = st.sidebar.time_input('Recording time', value=datetime.now().time())
    st.sidebar.caption('BirdNET uses date/time to filter by season.')
else:
    rec_date = None
    rec_time = None
    st.sidebar.caption('No date filter — all species considered.')

st.sidebar.subheader('Detection Settings')
conf_threshold = st.sidebar.slider('Confidence Threshold', 0.1, 1.0, 0.6)
st.sidebar.caption('Lower = more detections but more false positives.')

st.sidebar.markdown('---')
st.sidebar.info(
    '**Web demo** — BirdNET only.\n\n'
    'Perch classification and acoustic similarity search are available '
    'in the full desktop version.')

st.sidebar.markdown('---')
st.sidebar.header('Support')
st.sidebar.markdown(
    'For help contact:\n\n**Mike Shewring**\n\n'
    '[mike.shewring@rspb.org.uk](mailto:mike.shewring@rspb.org.uk)')


tab_analysis, tab_validation, tab_batch, tab_results = st.tabs([
    '🎧 Analysis', '✅ Validate', '⚙️ Batch Demo', '📊 Results',
])


# =============================================================================
# Analysis tab
# =============================================================================

with tab_analysis:

    uploaded_files = st.file_uploader(
        f'Upload Audio (WAV/MP3/FLAC/OGG — max {MAX_FILE_MB} MB per file)',
        accept_multiple_files=True, type=['wav', 'mp3', 'flac', 'ogg'])

    if uploaded_files:

        valid_files = []
        for f in uploaded_files:
            if f.size > MAX_FILE_MB * 1024 * 1024:
                st.warning(f"'{f.name}' exceeds {MAX_FILE_MB} MB and will be skipped.")
            else:
                valid_files.append(f)

        if not valid_files:
            st.error('No valid files to process.')
            st.stop()

        st.subheader('File Preview')
        selected_prev = st.selectbox('Select a file to preview:',
                                     [f.name for f in valid_files])
        preview_file = next(f for f in valid_files if f.name == selected_prev)

        col1, col2 = st.columns([1, 3])
        with col1:
            preview_file.seek(0)
            st.audio(preview_file)
        with col2:
            try:
                preview_file.seek(0)
                y_full, sr_full = librosa.load(preview_file, sr=PREVIEW_SR,
                                               duration=MAX_ANALYSIS_SEC,
                                               res_type='kaiser_fast')
                total_dur = librosa.get_duration(y=y_full, sr=sr_full)
                window_size = st.select_slider('Sonogram window size (s):',
                                               options=[30, 60, 120, 300], value=60)
                max_start = max(0.0, total_dur - window_size)
                win_start = (
                    st.slider('Scroll through recording:', min_value=0.0,
                              max_value=float(int(max_start)),
                              value=0.0, step=float(window_size // 2), format='%.0fs')
                    if max_start > 0 else 0.0)
                win_end = min(win_start + window_size, total_dur)

                y_slice = y_full[int(win_start * sr_full) : int(win_end * sr_full)]
                fig_p, ax_p = plt.subplots(figsize=(14, 3))
                librosa.display.specshow(_mel_db(y_slice, sr_full), x_axis='time',
                                         y_axis='mel', sr=sr_full, ax=ax_p, cmap='magma')
                ax_p.set_title(
                    f'Sonogram: {selected_prev}  '
                    f'[{win_start:.0f}s–{win_end:.0f}s / {total_dur:.0f}s]', fontsize=10)
                ax_p.set_xlabel('Time in window (s)')
                ax_p.set_ylabel('Frequency (mel)')
                fig_p.tight_layout()
                st.pyplot(fig_p)
                plt.close(fig_p)
                st.caption(f'Showing first {total_dur:.1f} s of the recording.')
            except Exception as e:
                st.warning(f'Could not generate sonogram: {e}')

        if st.button('🚀 Run BirdNET Analysis', key='run_analysis', type='primary'):
            all_detections = []
            audio_cache = {}
            progress = st.progress(0)
            status_msg = st.empty()

            for i, uf in enumerate(valid_files):
                status_msg.info(f'Analysing {uf.name}  ({i+1}/{len(valid_files)})')
                tmp_path = write_temp_upload(uf)

                try:
                    dets = run_birdnet(tmp_path, lat, lon, conf_threshold,
                                       use_date, rec_date, rec_time)
                    for d in dets:
                        d['file_name'] = uf.name
                        d['model'] = 'BirdNET'
                    all_detections.extend(dets)
                    if not dets:
                        st.info(f'No detections above threshold in {uf.name}')

                    y_audio, sr_audio = librosa.load(tmp_path, sr=PREVIEW_SR,
                                                     duration=MAX_ANALYSIS_SEC,
                                                     res_type='kaiser_fast')
                    audio_cache[uf.name] = (y_audio, sr_audio)

                except Exception as e:
                    st.error(f'Error processing {uf.name}: {e}')
                finally:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

                progress.progress((i + 1) / len(valid_files))

            status_msg.empty()

            st.session_state['all_detections'] = all_detections
            st.session_state['audio_cache'] = audio_cache
            st.session_state['valid_file_names'] = [f.name for f in valid_files]

            for d in all_detections:
                key = f"{d['file_name']}|{d['start_time']}"
                if key not in vstate():
                    set_vstatus(key, STATUS_UNREVIEWED)

    if st.session_state.get('all_detections'):

        all_detections = st.session_state['all_detections']
        audio_cache = st.session_state['audio_cache']
        valid_file_names = st.session_state['valid_file_names']

        df = pd.DataFrame(all_detections)
        df['validation'] = df.apply(
            lambda r: get_vstatus(f"{r['file_name']}|{r['start_time']}"), axis=1)

        st.success(f'Analysis complete: {len(df)} detections '
                   f'across {len(valid_file_names)} file(s).')

        st.subheader('Top Species by Confidence')
        top_sp = df.groupby('common_name')['confidence'].max() \
                   .sort_values(ascending=False).head(5)
        for col, (species, conf) in zip(st.columns(min(5, len(top_sp))), top_sp.items()):
            col.metric(label=species, value=f'{conf:.0%}',
                       delta=f'{len(df[df["common_name"] == species])} detection(s)')

        st.subheader('Sonogram Explorer')
        st.caption('Select a file and species to overlay detections. '
                   'Scroll through the recording with the window slider.')

        ca, cb = st.columns([1, 3])
        with ca:
            sono_file = st.selectbox('Select recording:',
                                     options=list(audio_cache.keys()), key='sono_file')
        with cb:
            file_sp = sorted(df[df['file_name'] == sono_file]['common_name'].unique())
            if file_sp:
                sel_sp = st.multiselect('Species to overlay:', options=file_sp,
                                        default=file_sp, key='sono_species')
            else:
                sel_sp = []
                st.info('No detections in this file.')

        min_conf_sono = st.slider('Minimum confidence to show:', 0.1, 1.0,
                                  value=conf_threshold, step=0.05, key='sono_conf')

        if sono_file in audio_cache:
            y_sono, sr_sono = audio_cache[sono_file]
            total_sono = librosa.get_duration(y=y_sono, sr=sr_sono)
            sono_win = st.select_slider('Window size (s):', options=[30, 60, 120, 300],
                                        value=60, key='sono_window')
            max_s_sono = max(0.0, total_sono - sono_win)
            sono_start = (
                st.slider('Scroll through recording:', min_value=0.0,
                          max_value=float(int(max_s_sono)), value=0.0,
                          step=float(sono_win // 2), format='%.0fs', key='sono_scroll')
                if max_s_sono > 0 else 0.0)
            sono_end = min(sono_start + sono_win, total_sono)

            visible_dets = ([d for d in all_detections
                             if d['file_name'] == sono_file
                             and d['common_name'] in sel_sp
                             and d['confidence'] >= min_conf_sono] if sel_sp else [])

            fig_sono = plot_annotated_sonogram(y_sono, sr_sono, visible_dets, sono_file,
                                               t_start_sec=sono_start, t_end_sec=sono_end)
            st.pyplot(fig_sono)
            plt.close(fig_sono)

            in_win = [d for d in visible_dets
                      if d['end_time'] >= sono_start and d['start_time'] <= sono_end]
            st.caption(f'Showing {len(in_win)} detection(s) in window. '
                       f'Analysed duration: {total_sono:.1f} s')

        st.subheader('Detection Timeline')
        tl_fig = px.scatter(
            df, x='start_time', y='common_name', size='confidence', color='common_name',
            facet_col='file_name' if len(valid_file_names) > 1 else None, facet_col_wrap=2,
            hover_data={'file_name': True, 'confidence': ':.2f',
                        'start_time': ':.1f', 'end_time': ':.1f', 'common_name': False},
            labels={'start_time': 'Time from Recording Start (s)', 'common_name': 'Species'},
            title='Species Detections — Time from Recording Start',
            height=max(400, df['common_name'].nunique() * 45))
        tl_fig.update_layout(showlegend=False,
                             yaxis={'categoryorder': 'total ascending'},
                             xaxis={'showgrid': True},
                             plot_bgcolor='rgba(0,0,0,0)',
                             paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(tl_fig, use_container_width=True)

        st.subheader('Confidence Distribution by Species')
        box_fig = px.box(df, x='confidence', y='common_name', color='common_name',
                         labels={'confidence': 'Confidence', 'common_name': 'Species'},
                         height=max(400, df['common_name'].nunique() * 35))
        box_fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(box_fig, use_container_width=True)

        st.subheader('Raw Detections')
        sp_filt = st.multiselect('Filter by species:',
                                 options=sorted(df['common_name'].unique()),
                                 default=sorted(df['common_name'].unique()))
        filtered = df[df['common_name'].isin(sp_filt)]

        st.dataframe(
            filtered.style.background_gradient(subset=['confidence'], cmap='RdYlGn'),
            use_container_width=True)

        st.download_button('⬇️ Download Detections CSV',
                           data=filtered.to_csv(index=False).encode('utf-8'),
                           file_name='birdnet_detections.csv', mime='text/csv')

    elif 'all_detections' in st.session_state and not st.session_state['all_detections']:
        st.warning('No detections found. Try lowering the confidence threshold.')


# =============================================================================
# Validate tab
# =============================================================================

with tab_validation:

    st.subheader('✅ Validate Detections')

    all_dets = st.session_state.get('all_detections', [])
    a_cache = st.session_state.get('audio_cache', {})

    if not all_dets:
        st.info('Run an analysis in the Analysis tab first.')
    else:
        df_v = pd.DataFrame(all_dets)
        df_v['validation'] = df_v.apply(
            lambda r: get_vstatus(f"{r['file_name']}|{r['start_time']}"), axis=1)

        vc1, vc2 = st.columns(2)
        with vc1:
            val_sp_filt = st.multiselect('Show species:',
                                         options=sorted(df_v['common_name'].unique()),
                                         default=sorted(df_v['common_name'].unique()),
                                         key='val_species')
        with vc2:
            val_st_filt = st.multiselect(
                'Show status:',
                options=[STATUS_UNREVIEWED, STATUS_ACCEPTED, STATUS_REJECTED, STATUS_UNSURE],
                default=[STATUS_UNREVIEWED, STATUS_ACCEPTED, STATUS_REJECTED, STATUS_UNSURE],
                key='val_status')

        val_counts = df_v['validation'].value_counts()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Total', len(df_v))
        m2.metric('Accepted', int(val_counts.get(STATUS_ACCEPTED, 0)))
        m3.metric('Rejected', int(val_counts.get(STATUS_REJECTED, 0)))
        m4.metric('Unreviewed', int(val_counts.get(STATUS_UNREVIEWED, 0)))

        queue = df_v[df_v['common_name'].isin(val_sp_filt) &
                     df_v['validation'].isin(val_st_filt)].reset_index(drop=True)

        if queue.empty:
            st.success('No detections match the selected filters.')
        else:
            if 'val_idx' not in st.session_state:
                st.session_state['val_idx'] = 0
            idx = max(0, min(st.session_state['val_idx'], len(queue) - 1))

            n1, n2, n3 = st.columns([1, 6, 1])
            with n1:
                if st.button('◀ Prev', key='val_prev') and idx > 0:
                    st.session_state['val_idx'] = idx - 1
                    st.rerun()
            with n3:
                if st.button('Next ▶', key='val_next') and idx < len(queue) - 1:
                    st.session_state['val_idx'] = idx + 1
                    st.rerun()
            with n2:
                st.progress((idx + 1) / len(queue), text=f'{idx + 1} / {len(queue)}')

            det = queue.iloc[idx].to_dict()
            det_key = f"{det['file_name']}|{det['start_time']}"
            status = get_vstatus(det_key)

            left_col, right_col = st.columns([1, 2])

            with left_col:
                st.markdown(f"**{det['common_name']}**  `{det['confidence']:.0%}`")
                st.markdown(f"{det['start_time']:.1f}s – {det['end_time']:.1f}s  "
                            f"|  _{det['file_name']}_")
                st.markdown(f'Status: **{status}**')

                b1, b2, b3, b4 = st.columns(4)
                if b1.button('✅', key=f'acc_{det_key}', help='Accept'):
                    set_vstatus(det_key, STATUS_ACCEPTED)
                    st.session_state['val_idx'] = min(idx + 1, len(queue) - 1)
                    st.rerun()
                if b2.button('❌', key=f'rej_{det_key}', help='Reject'):
                    set_vstatus(det_key, STATUS_REJECTED)
                    st.session_state['val_idx'] = min(idx + 1, len(queue) - 1)
                    st.rerun()
                if b3.button('❓', key=f'uns_{det_key}', help='Unsure'):
                    set_vstatus(det_key, STATUS_UNSURE)
                    st.session_state['val_idx'] = min(idx + 1, len(queue) - 1)
                    st.rerun()
                if b4.button('⏭', key=f'skp_{det_key}', help='Skip'):
                    st.session_state['val_idx'] = min(idx + 1, len(queue) - 1)
                    st.rerun()

            with right_col:
                fname = det['file_name']
                if fname in a_cache:
                    y_val, sr_val = a_cache[fname]
                    try:
                        total_val = librosa.get_duration(y=y_val, sr=sr_val)
                        t_s = max(det['start_time'] - 1.0, 0)
                        t_e = min(det['end_time'] + 1.0, total_val)
                        y_snip = y_val[int(t_s * sr_val) : int(t_e * sr_val)]
                        ab = io.BytesIO()
                        sf.write(ab, y_snip, sr_val, format='WAV')
                        ab.seek(0)
                        st.audio(ab, format='audio/wav')
                    except Exception:
                        st.caption('Could not render audio.')
                    try:
                        fig_v, _ = plot_detection_sonogram(y_val, sr_val, det)
                        st.pyplot(fig_v)
                        plt.close(fig_v)
                    except Exception:
                        st.caption('Could not render spectrogram.')

        st.markdown('---')
        df_val = pd.DataFrame(all_dets)
        df_val['validation'] = df_val.apply(
            lambda r: get_vstatus(f"{r['file_name']}|{r['start_time']}"), axis=1)
        st.download_button('⬇️ Download Validated CSV',
                           data=df_val.to_csv(index=False).encode('utf-8'),
                           file_name='birdnet_detections_validated.csv', mime='text/csv')


# =============================================================================
# Batch demo tab
# =============================================================================

with tab_batch:

    st.subheader('⚙️ Batch Demo')
    st.caption(f'Demonstrates headless processing of multiple recordings. '
               f'Limited to {MAX_BATCH_FILES} files in this web build.')
    st.warning('Results in this demo are stored temporarily and are cleared '
               'whenever the app restarts.', icon='⚠️')

    batch_id = st.text_input(
        'Batch ID', value=f"batch_{datetime.today().strftime('%Y%m%d_%H%M')}",
        help='Unique label, e.g. SlieveBeagh_20260510.')

    batch_files = st.file_uploader('Upload batch audio files',
                                   accept_multiple_files=True,
                                   type=['wav', 'mp3', 'flac', 'ogg'],
                                   key='batch_upload')

    if batch_files:
        if len(batch_files) > MAX_BATCH_FILES:
            st.warning(f'Only the first {MAX_BATCH_FILES} files will be processed.')
            batch_files = batch_files[:MAX_BATCH_FILES]

        st.info(f'{len(batch_files)} files queued → batch **{batch_id}**')

        if st.button('▶️ Run Batch', key='run_batch', type='primary'):
            progress = st.progress(0)
            status_msg = st.empty()
            error_log = []
            det_count = 0

            for i, uf in enumerate(batch_files):
                if uf.size > MAX_FILE_MB * 1024 * 1024:
                    error_log.append(f'{uf.name}: exceeds {MAX_FILE_MB} MB, skipped')
                    progress.progress((i + 1) / len(batch_files))
                    continue

                status_msg.write(f'[{i+1}/{len(batch_files)}] {uf.name}…')
                tmp_path = write_temp_upload(uf)

                try:
                    dets = run_birdnet(tmp_path, lat, lon, conf_threshold,
                                       use_date, rec_date, rec_time)
                    if dets:
                        det_df = pd.DataFrame(dets)
                        det_df['file_name'] = uf.name
                        det_df['lat'] = lat
                        det_df['lon'] = lon
                        det_df['rec_date'] = str(rec_date) if use_date and rec_date else ''
                        det_df['source'] = 'birdnet'
                        det_df['verified'] = STATUS_UNREVIEWED
                        det_df['notes'] = ''
                        keep = ['file_name', 'start_time', 'end_time', 'common_name',
                                'scientific_name', 'confidence', 'lat', 'lon',
                                'rec_date', 'source', 'verified', 'notes']
                        save_detections_db(batch_id,
                                           det_df[[c for c in keep if c in det_df.columns]])
                        det_count += len(dets)
                except Exception as e:
                    error_log.append(f'{uf.name}: {e}')
                finally:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

                progress.progress((i + 1) / len(batch_files))

            status_msg.empty()
            n_ok = len(batch_files) - len(error_log)
            st.success(f'✅ Batch complete — {det_count} detections '
                       f'from {n_ok} / {len(batch_files)} files.')
            if error_log:
                with st.expander(f'⚠️ {len(error_log)} issues'):
                    for err in error_log:
                        st.code(err)
            st.info('Switch to the **Results** tab to review.')


# =============================================================================
# Results tab
# =============================================================================

with tab_results:

    st.subheader('📊 Results')

    all_db = load_detections_db()

    if all_db.empty:
        st.info('No batch results yet. Run a batch in the Batch Demo tab.')
    else:
        batches = sorted(all_db['batch_id'].unique(), reverse=True)
        sel_batch_r = st.selectbox('Batch:', ['All'] + list(batches), key='res_batch')

        view_df = (all_db[all_db['batch_id'] == sel_batch_r].copy()
                   if sel_batch_r != 'All' else all_db.copy())

        st.caption(f"{len(view_df)} detections  |  "
                   f"{view_df['file_name'].nunique()} files  |  "
                   f"{view_df['common_name'].nunique()} species")

        rc1, rc2 = st.columns(2)
        with rc1:
            r_sp = st.multiselect('Species:',
                                  sorted(view_df['common_name'].unique()), default=[])
        with rc2:
            r_st = st.multiselect(
                'Status:',
                [STATUS_UNREVIEWED, STATUS_ACCEPTED, STATUS_REJECTED, STATUS_UNSURE],
                default=[STATUS_UNREVIEWED, STATUS_ACCEPTED])

        if r_sp:
            view_df = view_df[view_df['common_name'].isin(r_sp)]
        if r_st:
            view_df = view_df[view_df['verified'].isin(r_st)]

        display_cols = [c for c in ['batch_id', 'file_name', 'start_time', 'end_time',
                                    'common_name', 'confidence', 'verified', 'notes']
                        if c in view_df.columns]
        st.dataframe(
            view_df[display_cols].style.background_gradient(
                subset=['confidence'], cmap='RdYlGn'),
            use_container_width=True, hide_index=True)

        with st.expander('📈 Summary charts'):
            pc1, pc2 = st.columns(2)
            with pc1:
                top20 = (view_df.groupby('common_name').size().reset_index(name='count')
                         .sort_values('count', ascending=False).head(20))
                st.plotly_chart(px.bar(top20, x='count', y='common_name',
                                       orientation='h', title='Top 20 species'),
                                use_container_width=True)
            with pc2:
                st.plotly_chart(
                    px.pie(view_df, names='verified', title='Verification status',
                           color='verified',
                           color_discrete_map={
                               STATUS_ACCEPTED: '#2ca02c',
                               STATUS_REJECTED: '#d62728',
                               STATUS_UNREVIEWED: '#aec7e8',
                               STATUS_UNSURE: '#ff7f0e'}),
                    use_container_width=True)

        st.download_button('⬇️ Export filtered results as CSV',
                           data=view_df.to_csv(index=False).encode('utf-8'),
                           file_name=f'results_{sel_batch_r}.csv', mime='text/csv')
