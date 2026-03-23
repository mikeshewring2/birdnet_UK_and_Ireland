################################## Audio Analyzer App ##################################
#
# A Streamlit app for analyzing audio files with BirdNET and Perch, visualizing detections,
# and validating results. Detections can be saved to a local SQLite database for later review
# and batch processing.
#
# Developed by Mike Shewring / RSPB Centre for Conservation Science.
# mike.shewring@rspb.org.uk
#
# Requirements:
#   pip install streamlit librosa matplotlib plotly soundfile birdnetlib
#   Optional for Perch: pip install tensorflow tensorflow_hub bioacoustics-model-zoo
#   Run with: streamlit run app2.py
#
# to do - add Perch embedding viewer + "find similar" function
# to do - add option to save clips of detections    


import streamlit as st
import os
import io
import tempfile
import base64
import sqlite3
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import librosa
import librosa.display
import plotly.express as px
import soundfile as sf
from scipy.spatial.distance import cdist

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

try:
    import bioacoustics_model_zoo as bmz
    import tensorflow as _tf        # noqa: F401
    import tensorflow_hub as _tfhub # noqa: F401
    PERCH_AVAILABLE = True
    PERCH_ERROR = None
except Exception as e:
    PERCH_AVAILABLE = False
    PERCH_ERROR = str(e)


MAX_FILE_MB = 500
DB_PATH = Path('birdnet_results.db')
PERCH_WINDOW_SEC = 5.0

STATUS_UNREVIEWED = 'Unreviewed'
STATUS_ACCEPTED = 'Accepted'
STATUS_REJECTED = 'Rejected'
STATUS_UNSURE = 'Unsure'


# --- database helpers ---

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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id     TEXT,
            file_name    TEXT,
            window_start REAL,
            embedding    BLOB
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


def save_embeddings_db(batch_id, file_name, window_starts, emb_array):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    rows = [
        (batch_id, file_name, float(ws), emb.astype(np.float32).tobytes())
        for ws, emb in zip(window_starts, emb_array)
    ]
    cur.executemany(
        'INSERT INTO embeddings (batch_id, file_name, window_start, embedding) VALUES (?,?,?,?)',
        rows
    )
    con.commit()
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
    con.execute(
        'UPDATE detections SET verified=?, notes=? WHERE id=?',
        (status, notes, det_id)
    )
    con.commit()
    con.close()


def insert_detection_db(batch_id, file_name, start_time, end_time,
                        common_name, scientific_name, confidence,
                        lat, lon, rec_date, source, verified, notes=''):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO detections
            (batch_id, file_name, start_time, end_time, common_name,
             scientific_name, confidence, lat, lon, rec_date, source, verified, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (batch_id, file_name, start_time, end_time,
          common_name, scientific_name, confidence,
          lat, lon, rec_date, source, verified, notes))
    con.commit()
    con.close()


def get_batch_ids():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute('SELECT DISTINCT batch_id FROM detections ORDER BY batch_id DESC')
    ids = [r[0] for r in cur.fetchall()]
    con.close()
    return ids


# --- model loading ---

@st.cache_resource
def load_analyzer():
    return Analyzer()


@st.cache_resource
def load_perch():
    return bmz.Perch() if PERCH_AVAILABLE else None


# --- spectrogram helpers ---

def plot_annotated_sonogram(y, sr, detections, filename, t_start_sec=0, t_end_sec=60):
    s_start = int(t_start_sec * sr)
    s_end = int(t_end_sec * sr)
    y_slice = y[s_start : min(s_end, len(y))]

    fig, ax = plt.subplots(figsize=(14, 4))
    S = librosa.feature.melspectrogram(y=y_slice, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, ax=ax, cmap='magma')

    window_dets = [d for d in detections
                   if d['end_time'] >= t_start_sec and d['start_time'] <= t_end_sec]
    species_list = list(set(d['common_name'] for d in window_dets))
    cmap = {sp: plt.cm.tab20(i / max(len(species_list), 1)) for i, sp in enumerate(species_list)}

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
    S_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y_snip, sr=sr, n_mels=128), ref=np.max)
    librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, ax=ax, cmap='magma')
    # shade the actual detection window
    ax.axvspan(pad, pad + (det['end_time'] - det['start_time']), alpha=0.3, color='white')
    ax.set_title(f"{det['common_name']}  {det['start_time']:.1f}s–{det['end_time']:.1f}s",
                 fontsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.tight_layout()
    return fig, y_snip


def clip_to_b64(file_path, start_sec, end_sec, sr=32000):
    y, _ = librosa.load(file_path, sr=sr, offset=start_sec, duration=(end_sec - start_sec))
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV')
    buf.seek(0)
    return f'data:audio/wav;base64,{base64.b64encode(buf.read()).decode()}'


def inline_player(file_path, start_sec, end_sec):
    try:
        uri = clip_to_b64(file_path, start_sec, end_sec)
        st.markdown(f'<audio controls style="width:100%;" src="{uri}"></audio>',
                    unsafe_allow_html=True)
    except Exception as e:
        st.caption(f'Could not extract clip: {e}')


# --- analysis ---

def run_birdnet(file_path, lat, lon, conf, use_date, rec_date, rec_time):
    analyzer = load_analyzer()
    kwargs = dict(lat=lat, lon=lon, min_conf=conf)
    if use_date and rec_date:
        kwargs['date'] = datetime.combine(rec_date, rec_time)
    recording = Recording(analyzer, str(file_path), **kwargs)
    recording.analyze()
    return recording.detections


def run_perch_embeddings(file_path):
    perch = load_perch()
    if perch is None:
        return None, None
    emb_df = perch.embed([str(file_path)])
    window_starts = emb_df.index.get_level_values('start_time').values
    return window_starts, emb_df.values.astype(np.float32)


def run_perch_classify(file_path, top_n=10):
    """Perch classifier — scores softmax-normalised per window to [0,1]."""
    from scipy.special import softmax as _softmax

    perch = load_perch()
    if perch is None:
        return []

    scores_df = perch.predict([str(file_path)])
    detections = []

    for row_idx, row in scores_df.iterrows():
        if isinstance(row_idx, tuple):
            start = float(row_idx[1]) if len(row_idx) > 1 else 0.0
            end = float(row_idx[2]) if len(row_idx) > 2 else start + 5.0
        else:
            start, end = 0.0, 5.0

        norm = _softmax(row.values)
        for i in np.argsort(norm)[::-1][:top_n]:
            score = float(norm[i])
            if score < 0.01:
                break
            detections.append({
                'common_name': str(row.index[i]),
                'scientific_name': '',
                'confidence': score,
                'start_time': start,
                'end_time': end,
                'model': 'Perch',
            })

    return detections


def find_similar(query_start, window_starts, emb_matrix, top_n=10):
    query_idx = int(np.argmin(np.abs(window_starts - query_start)))
    distances = cdist(emb_matrix[query_idx:query_idx+1], emb_matrix, metric='cosine')[0]
    idx = np.argsort(distances)
    idx = idx[idx != query_idx][:top_n]
    return pd.DataFrame({
        'window_start': window_starts[idx],
        'window_end': window_starts[idx] + PERCH_WINDOW_SEC,
        'cosine_distance': distances[idx],
        'similarity_pct': (1 - distances[idx]) * 100,
    }).reset_index(drop=True)


# --- validation state helpers ---

def vstate():
    if 'validation' not in st.session_state:
        st.session_state['validation'] = {}
    return st.session_state['validation']

def get_vstatus(key):
    return vstate().get(key, STATUS_UNREVIEWED)

def set_vstatus(key, status):
    st.session_state['validation'][key] = status


# --- app startup ---

init_db()

st.set_page_config(page_title='Audio Analyzer', layout='wide')
st.title('🦆 Audio Analyzer 🦉')


# sidebar

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

st.sidebar.subheader('Model Selection')
if PERCH_AVAILABLE:
    model_choice = st.sidebar.radio(
        'Classifier',
        ['BirdNET', 'Perch', 'Both'],
        index=0,
        help=(
            'BirdNET: fast, 6000+ species, geographic/seasonal filter.\n'
            'Perch: ~10,000 species, global model, softmax-normalised scores.\n'
            'Both: run in parallel — results tagged by model.'
        )
    )
    use_perch = False  # Find Similar not yet wired up
else:
    model_choice = 'BirdNET'
    use_perch = False
    st.sidebar.warning(
        f'Perch unavailable — {PERCH_ERROR}\n\n'
        'Install with: `pip install tensorflow tensorflow_hub bioacoustics-model-zoo`'
    )

st.sidebar.markdown('---')
st.sidebar.header('Support')
st.sidebar.markdown(
    'For help contact:\n\n**Mike Shewring**\n\n'
    '[mike.shewring@rspb.org.uk](mailto:mike.shewring@rspb.org.uk)')


tab_analysis, tab_validation, tab_batch, tab_results = st.tabs([
    '🎧 Analysis', '✅ Validate', '⚙️ Batch Pipeline', '📊 Pipeline Results',
])


# =============================================================================
# Analysis tab
# =============================================================================

with tab_analysis:

    with st.expander('💾 Save session to database (optional)'):
        save_to_db = st.checkbox('Save results to pipeline database', value=False)
        session_batch_id = st.text_input(
            'Batch ID for this session',
            value=f"session_{datetime.today().strftime('%Y%m%d_%H%M')}",
            disabled=not save_to_db)

    uploaded_files = st.file_uploader(
        'Upload Audio (WAV/MP3)', accept_multiple_files=True,
        type=['wav', 'mp3', 'flac', 'ogg'])

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
        selected_prev = st.selectbox('Select a file to preview:', [f.name for f in valid_files])
        preview_file = next(f for f in valid_files if f.name == selected_prev)

        col1, col2 = st.columns([1, 3])
        with col1:
            preview_file.seek(0)
            st.audio(preview_file)
        with col2:
            try:
                preview_file.seek(0)
                y_full, sr_full = librosa.load(preview_file, sr=None)
                total_dur = librosa.get_duration(y=y_full, sr=sr_full)
                window_size = st.select_slider('Sonogram window size (s):',
                                               options=[30, 60, 120, 300], value=60)
                max_start = max(0.0, total_dur - window_size)
                win_start = (
                    st.slider('Scroll through recording:',
                              min_value=0.0,
                              max_value=float(int(max_start)) if max_start > 0 else 0.0,
                              value=0.0, step=float(window_size // 2), format='%.0fs')
                    if max_start > 0 else 0.0)
                win_end = min(win_start + window_size, total_dur)

                y_slice = y_full[int(win_start * sr_full) : int(win_end * sr_full)]
                fig_p, ax_p = plt.subplots(figsize=(14, 3))
                S_db = librosa.power_to_db(
                    librosa.feature.melspectrogram(y=y_slice, sr=sr_full, n_mels=128),
                    ref=np.max)
                librosa.display.specshow(S_db, x_axis='time', y_axis='mel',
                                         sr=sr_full, ax=ax_p, cmap='magma')
                ax_p.set_title(
                    f'Sonogram: {selected_prev}  [{win_start:.0f}s–{win_end:.0f}s / {total_dur:.0f}s]',
                    fontsize=10)
                ax_p.set_xlabel('Time in window (s)')
                ax_p.set_ylabel('Frequency (mel)')
                fig_p.tight_layout()
                st.pyplot(fig_p)
                plt.close(fig_p)
                st.caption(f'Total recording duration: {total_dur:.1f} s')
            except Exception as e:
                st.warning(f'Could not generate sonogram: {e}')

        run_label = {
            'BirdNET': '🚀 Run BirdNET Analysis',
            'Perch': '🚀 Run Perch Analysis',
            'Both': '🚀 Run BirdNET + Perch Analysis',
        }.get(model_choice, '🚀 Run Analysis')

        if st.button(run_label, key='run_analysis'):
            all_detections = []
            audio_cache = {}
            emb_cache = {}
            progress = st.progress(0)
            status_msg = st.empty()

            for i, uf in enumerate(valid_files):
                status_msg.info(f'Analysing {uf.name}  ({i+1}/{len(valid_files)})')
                suffix = os.path.splitext(uf.name)[1]

                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uf.getbuffer())
                    tmp_path = tmp.name

                try:
                    if model_choice in ('BirdNET', 'Both'):
                        dets_bn = run_birdnet(tmp_path, lat, lon, conf_threshold,
                                              use_date, rec_date, rec_time)
                        for d in dets_bn:
                            d['file_name'] = uf.name
                            d['model'] = 'BirdNET'
                        all_detections.extend(dets_bn)
                        if not dets_bn:
                            st.info(f'No BirdNET detections above threshold in {uf.name}')

                    if model_choice in ('Perch', 'Both'):
                        status_msg.info(f'Running Perch classifier on {uf.name}…')
                        try:
                            dets_p = run_perch_classify(tmp_path, top_n=10)
                            for d in dets_p:
                                d['file_name'] = uf.name
                            all_detections.extend(dets_p)
                            if not dets_p:
                                st.info(f'No Perch detections in {uf.name}')
                        except Exception as e:
                            st.warning(f'Perch classification failed for {uf.name}: {e}')

                    uf.seek(0)
                    y_audio, sr_audio = librosa.load(tmp_path, sr=None)
                    audio_cache[uf.name] = (y_audio, sr_audio)

                except Exception as e:
                    st.error(f'Error processing {uf.name}: {e}')
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)

                progress.progress((i + 1) / len(valid_files))

            status_msg.empty()

            st.session_state['all_detections'] = all_detections
            st.session_state['audio_cache'] = audio_cache
            st.session_state['emb_cache'] = emb_cache
            st.session_state['valid_file_names'] = [f.name for f in valid_files]

            if all_detections:
                for d in all_detections:
                    key = f"{d['file_name']}|{d['start_time']}"
                    if key not in vstate():
                        set_vstatus(key, STATUS_UNREVIEWED)

            if save_to_db and all_detections:
                det_df = pd.DataFrame(all_detections)
                det_df['lat'] = lat
                det_df['lon'] = lon
                det_df['rec_date'] = str(rec_date) if rec_date else ''
                det_df['source'] = 'birdnet'
                det_df['verified'] = STATUS_UNREVIEWED
                det_df['notes'] = ''
                save_detections_db(session_batch_id, det_df)
                st.success(f"Saved to database under batch '{session_batch_id}'.")

    if st.session_state.get('all_detections'):

        all_detections = st.session_state['all_detections']
        audio_cache = st.session_state['audio_cache']
        emb_cache = st.session_state.get('emb_cache', {})
        valid_file_names = st.session_state['valid_file_names']

        df = pd.DataFrame(all_detections)
        df['validation'] = df.apply(
            lambda r: get_vstatus(f"{r['file_name']}|{r['start_time']}"), axis=1)

        st.success(f'Analysis complete: {len(df)} detections across {len(valid_file_names)} file(s).')

        st.subheader('Top Species by Confidence')
        top_sp = df.groupby('common_name')['confidence'].max().sort_values(ascending=False).head(5)
        for col, (species, conf) in zip(st.columns(min(5, len(top_sp))), top_sp.items()):
            col.metric(label=species, value=f'{conf:.0%}',
                       delta=f'{len(df[df["common_name"] == species])} detection(s)')

        st.subheader('Sonogram Explorer')
        st.caption('Select a file and species to overlay detections. '
                   'Scroll through long recordings with the window slider.')

        ca, cb, cc = st.columns([1, 2, 1])
        with ca:
            sono_file = st.selectbox('Select recording:',
                                     options=list(audio_cache.keys()), key='sono_file')
        with cc:
            available_models = sorted(df['model'].unique()) if 'model' in df.columns else ['BirdNET']
            sel_models = st.multiselect('Models to show:', options=available_models,
                                        default=available_models, key='sono_models')
        with cb:
            file_sp = sorted(
                df[(df['file_name'] == sono_file) &
                   (df['model'].isin(sel_models) if 'model' in df.columns else True)
                ]['common_name'].unique())
            if file_sp:
                sel_sp = st.multiselect('Species to overlay:', options=file_sp,
                                        default=file_sp, key='sono_species')
            else:
                sel_sp = []
                st.info('No detections in this file.')

        min_conf_sono = st.slider('Minimum confidence to show:',
                                  min_value=0.1, max_value=1.0,
                                  value=conf_threshold, step=0.05, key='sono_conf')

        if sono_file in audio_cache:
            y_sono, sr_sono = audio_cache[sono_file]
            total_sono = librosa.get_duration(y=y_sono, sr=sr_sono)
            sono_win = st.select_slider('Window size (s):', options=[30, 60, 120, 300],
                                        value=60, key='sono_window')
            max_s_sono = max(0.0, total_sono - sono_win)
            sono_start = (
                st.slider('Scroll through recording:',
                          min_value=0.0,
                          max_value=float(int(max_s_sono)) if max_s_sono > 0 else 0.0,
                          value=0.0, step=float(sono_win // 2),
                          format='%.0fs', key='sono_scroll')
                if max_s_sono > 0 else 0.0)
            sono_end = min(sono_start + sono_win, total_sono)

            visible_dets = (
                [d for d in all_detections
                 if d['file_name'] == sono_file
                 and d['common_name'] in sel_sp
                 and d.get('model', 'BirdNET') in sel_models
                 and d['confidence'] >= min_conf_sono]
                if sel_sp else [])

            fig_sono = plot_annotated_sonogram(y_sono, sr_sono, visible_dets, sono_file,
                                               t_start_sec=sono_start, t_end_sec=sono_end)
            st.pyplot(fig_sono)
            plt.close(fig_sono)

            in_win = [d for d in visible_dets
                      if d['end_time'] >= sono_start and d['start_time'] <= sono_end]
            st.caption(f'Showing {len(in_win)} detection(s) in window. '
                       f'Recording duration: {total_sono:.1f} s')

        st.subheader('Detection Timeline')
        tl_fig = px.scatter(
            df, x='start_time', y='common_name',
            size='confidence', color='common_name',
            facet_col='file_name' if len(valid_file_names) > 1 else None,
            facet_col_wrap=2,
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
        fi_filt = (
            st.multiselect('Filter by file:',
                           options=sorted(df['file_name'].unique()),
                           default=sorted(df['file_name'].unique()))
            if len(valid_file_names) > 1 else [valid_file_names[0]])
        model_filt = st.multiselect(
            'Filter by model:',
            options=sorted(df['model'].unique()) if 'model' in df.columns else [],
            default=sorted(df['model'].unique()) if 'model' in df.columns else [])

        filtered = df[df['common_name'].isin(sp_filt) & df['file_name'].isin(fi_filt)]
        if model_filt and 'model' in df.columns:
            filtered = filtered[filtered['model'].isin(model_filt)]

        st.dataframe(filtered.style.background_gradient(subset=['confidence'], cmap='RdYlGn'),
                     use_container_width=True)

        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            st.download_button('Download Filtered CSV',
                               data=filtered.to_csv(index=False).encode('utf-8'),
                               file_name='birdnet_detections_filtered.csv', mime='text/csv')
        with c_dl2:
            st.download_button('Download All Detections CSV',
                               data=df.to_csv(index=False).encode('utf-8'),
                               file_name='birdnet_detections_all.csv', mime='text/csv')

    elif 'all_detections' in st.session_state and not st.session_state['all_detections']:
        st.warning('No detections found. Try lowering the confidence threshold.')


# =============================================================================
# Validate tab
# =============================================================================

with tab_validation:

    st.subheader('✅ Validate Detections')

    val_source = st.radio('Validate from:', ['Current session', 'Pipeline database'],
                          horizontal=True)

    if val_source == 'Current session':

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
            m2.metric('Accepted', val_counts.get(STATUS_ACCEPTED, 0))
            m3.metric('Rejected', val_counts.get(STATUS_REJECTED, 0))
            m4.metric('Unreviewed', val_counts.get(STATUS_UNREVIEWED, 0))

            queue = df_v[
                df_v['common_name'].isin(val_sp_filt) &
                df_v['validation'].isin(val_st_filt)
            ].reset_index(drop=True)

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

                border = {
                    STATUS_ACCEPTED: '#2ecc71',
                    STATUS_REJECTED: '#e74c3c',
                    STATUS_UNSURE: '#f39c12',
                    STATUS_UNREVIEWED: '#888888',
                }.get(status, '#888888')

                st.markdown(
                    f'<div style="border-left:4px solid {border};padding-left:12px;margin-bottom:4px">',
                    unsafe_allow_html=True)

                left_col, right_col = st.columns([1, 2])

                with left_col:
                    st.markdown(f"**{det['common_name']}**  `{det['confidence']:.0%}`")
                    st.markdown(f"{det['start_time']:.1f}s – {det['end_time']:.1f}s  |  _{det['file_name']}_")
                    st.markdown(f'Status: **{status}**')

                    notes_val = st.text_input('Notes:', key=f'vnotes_{det_key}',
                                              placeholder='Add a note or corrected ID…')

                    b1, b2, b3, b4 = st.columns(4)
                    if b1.button('✅', key=f'acc_{det_key}', help='Accept'):
                        set_vstatus(det_key, STATUS_ACCEPTED)
                        if save_to_db:
                            update_verified_db(det.get('id', -1), STATUS_ACCEPTED, notes_val)
                        st.session_state['val_idx'] = min(idx + 1, len(queue) - 1)
                        st.rerun()
                    if b2.button('❌', key=f'rej_{det_key}', help='Reject'):
                        set_vstatus(det_key, STATUS_REJECTED)
                        if save_to_db:
                            update_verified_db(det.get('id', -1), STATUS_REJECTED, notes_val)
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

                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('---')
            df_val = pd.DataFrame(all_dets)
            df_val['validation'] = df_val.apply(
                lambda r: get_vstatus(f"{r['file_name']}|{r['start_time']}"), axis=1)
            st.download_button(
                'Download Validated CSV',
                data=df_val.to_csv(index=False).encode('utf-8'),
                file_name='birdnet_detections_validated.csv', mime='text/csv')

    else:
        batch_ids = get_batch_ids()
        if not batch_ids:
            st.info('No pipeline batches in database yet.')
        else:
            sel_batch = st.selectbox('Select batch:', batch_ids)
            db_df = load_detections_db(sel_batch)

            db_st_filt = st.multiselect(
                'Show status:',
                options=[STATUS_UNREVIEWED, STATUS_ACCEPTED, STATUS_REJECTED, STATUS_UNSURE],
                default=[STATUS_UNREVIEWED], key='db_st_filt')
            db_queue = db_df[db_df['verified'].isin(db_st_filt)].reset_index(drop=True)

            db_counts = db_df['verified'].value_counts()
            dm1, dm2, dm3, dm4 = st.columns(4)
            dm1.metric('Total', len(db_df))
            dm2.metric('Accepted', db_counts.get(STATUS_ACCEPTED, 0))
            dm3.metric('Rejected', db_counts.get(STATUS_REJECTED, 0))
            dm4.metric('Unreviewed', db_counts.get(STATUS_UNREVIEWED, 0))

            if db_queue.empty:
                st.success('No detections with the selected status.')
            else:
                st.caption(f'{len(db_queue)} detections in queue')

                if 'db_val_idx' not in st.session_state:
                    st.session_state['db_val_idx'] = 0
                db_idx = max(0, min(st.session_state['db_val_idx'], len(db_queue) - 1))

                dn1, dn2, dn3 = st.columns([1, 6, 1])
                with dn1:
                    if st.button('◀ Prev', key='db_prev') and db_idx > 0:
                        st.session_state['db_val_idx'] = db_idx - 1
                        st.rerun()
                with dn3:
                    if st.button('Next ▶', key='db_next') and db_idx < len(db_queue) - 1:
                        st.session_state['db_val_idx'] = db_idx + 1
                        st.rerun()
                with dn2:
                    st.progress((db_idx + 1) / len(db_queue),
                                text=f'{db_idx + 1} / {len(db_queue)}')

                db_row = db_queue.iloc[db_idx]
                st.markdown(
                    f"### {db_row['common_name']}  "
                    f"<small style='color:grey;'>confidence {db_row['confidence']:.2f}</small>",
                    unsafe_allow_html=True)
                st.caption(
                    f"**Batch:** {db_row['batch_id']}  |  **File:** {db_row['file_name']}  |  "
                    f"**Time:** {db_row['start_time']:.1f}s — {db_row['end_time']:.1f}s  |  "
                    f"**Source:** {db_row.get('source', 'birdnet')}")

                st.info('💡 Audio playback for pipeline results requires the original file to be present. '
                        'Re-upload via the Analysis tab if needed.')

                db_notes = st.text_input('Notes:', key=f"dbnotes_{db_row['id']}",
                                         value=str(db_row.get('notes', '') or ''),
                                         placeholder='Add a note or corrected ID…')

                da1, da2, da3, da4 = st.columns(4)
                with da1:
                    if st.button('✅ Accept', key=f"dba_{db_row['id']}", type='primary'):
                        update_verified_db(db_row['id'], STATUS_ACCEPTED, db_notes)
                        st.session_state['db_val_idx'] = min(db_idx + 1, len(db_queue) - 1)
                        st.rerun()
                with da2:
                    if st.button('❌ Reject', key=f"dbr_{db_row['id']}"):
                        update_verified_db(db_row['id'], STATUS_REJECTED, db_notes)
                        st.session_state['db_val_idx'] = min(db_idx + 1, len(db_queue) - 1)
                        st.rerun()
                with da3:
                    if st.button('❓ Unsure', key=f"dbu_{db_row['id']}"):
                        update_verified_db(db_row['id'], STATUS_UNSURE, db_notes)
                        st.session_state['db_val_idx'] = min(db_idx + 1, len(db_queue) - 1)
                        st.rerun()
                with da4:
                    if st.button('⏭ Skip', key=f"dbs_{db_row['id']}"):
                        st.session_state['db_val_idx'] = min(db_idx + 1, len(db_queue) - 1)
                        st.rerun()


# =============================================================================
# Batch pipeline tab
# =============================================================================

with tab_batch:

    st.subheader('⚙️ Batch Pipeline')
    st.caption('Process large batches of ARU recordings headlessly. '
               'All results saved to the persistent database.')

    batch_id = st.text_input(
        'Batch ID',
        value=f"batch_{datetime.today().strftime('%Y%m%d_%H%M')}",
        help='Unique label, e.g. SlieveBog_20260510.')

    batch_files = st.file_uploader(
        'Upload batch audio files', accept_multiple_files=True,
        type=['wav', 'mp3', 'flac', 'ogg'], key='batch_upload')

    with st.expander('Batch settings (overrides sidebar for this run)'):
        b_lat = st.number_input('Latitude', value=lat, format='%.4f', key='b_lat')
        b_lon = st.number_input('Longitude', value=lon, format='%.4f', key='b_lon')
        b_conf = st.slider('Confidence threshold', 0.1, 1.0, conf_threshold, key='b_conf')
        b_use_date = st.checkbox('Apply seasonal date filter', value=use_date, key='b_use_date')
        b_date = st.date_input('Recording date', value=date.today(), key='b_date')
        b_time = st.time_input('Recording time', value=datetime.now().time(), key='b_time')
        b_perch = st.checkbox(
            'Generate Perch embeddings (enables similarity search on results)',
            value=use_perch and PERCH_AVAILABLE)

    if batch_files:
        st.info(f'{len(batch_files)} files queued → batch **{batch_id}**')

        if st.button('▶️ Run Batch', key='run_batch'):
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

                with tempfile.NamedTemporaryFile(suffix=Path(uf.name).suffix, delete=False) as tmp:
                    tmp.write(uf.getbuffer())
                    tmp_path = tmp.name

                try:
                    dets = run_birdnet(tmp_path, b_lat, b_lon, b_conf,
                                       b_use_date, b_date, b_time)
                    if dets:
                        det_df = pd.DataFrame(dets)
                        det_df['file_name'] = uf.name
                        det_df['lat'] = b_lat
                        det_df['lon'] = b_lon
                        det_df['rec_date'] = str(b_date) if b_use_date else ''
                        det_df['source'] = 'birdnet'
                        det_df['verified'] = STATUS_UNREVIEWED
                        det_df['notes'] = ''
                        save_detections_db(batch_id, det_df)
                        det_count += len(dets)

                    if b_perch:
                        status_msg.write(f'  → Embeddings for {uf.name}…')
                        w_starts, emb_matrix = run_perch_embeddings(tmp_path)
                        if emb_matrix is not None:
                            save_embeddings_db(batch_id, uf.name, w_starts, emb_matrix)

                except Exception as e:
                    error_log.append(f'{uf.name}: {e}')
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

                progress.progress((i + 1) / len(batch_files))

            status_msg.empty()
            n_ok = len(batch_files) - len(error_log)
            st.success(f'✅ Batch complete — {det_count} detections from {n_ok} / {len(batch_files)} files.')
            if error_log:
                with st.expander(f'⚠️ {len(error_log)} issues'):
                    for err in error_log:
                        st.code(err)
            st.info('Switch to **Pipeline Results** or **Validate** to review.')


# =============================================================================
# Pipeline results tab
# =============================================================================

with tab_results:

    st.subheader('📊 Pipeline Results')

    all_db = load_detections_db()

    if all_db.empty:
        st.info('No pipeline results yet. Run a batch in the Batch Pipeline tab.')
    else:
        batches = sorted(all_db['batch_id'].unique(), reverse=True)
        sel_batch_r = st.selectbox('Batch:', ['All'] + list(batches), key='res_batch')

        view_df = (all_db[all_db['batch_id'] == sel_batch_r].copy()
                   if sel_batch_r != 'All' else all_db.copy())

        st.caption(f"{len(view_df)} detections  |  "
                   f"{view_df['file_name'].nunique()} files  |  "
                   f"{view_df['common_name'].nunique()} species")

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            r_sp = st.multiselect('Species:', sorted(view_df['common_name'].unique()), default=[])
        with rc2:
            r_st = st.multiselect('Status:',
                                  [STATUS_UNREVIEWED, STATUS_ACCEPTED, STATUS_REJECTED, STATUS_UNSURE],
                                  default=[STATUS_UNREVIEWED, STATUS_ACCEPTED])
        with rc3:
            r_src = st.multiselect(
                'Source:',
                sorted(view_df['source'].unique()) if 'source' in view_df.columns else [],
                default=[])

        if r_sp:
            view_df = view_df[view_df['common_name'].isin(r_sp)]
        if r_st:
            view_df = view_df[view_df['verified'].isin(r_st)]
        if r_src and 'source' in view_df.columns:
            view_df = view_df[view_df['source'].isin(r_src)]

        display_cols = [c for c in ['batch_id', 'file_name', 'start_time', 'end_time',
                                     'common_name', 'confidence', 'verified', 'source', 'notes']
                        if c in view_df.columns]
        st.dataframe(view_df[display_cols].style.background_gradient(
            subset=['confidence'], cmap='RdYlGn'),
            use_container_width=True, hide_index=True)

        with st.expander('📈 Summary charts'):
            pc1, pc2 = st.columns(2)
            with pc1:
                top20 = (view_df.groupby('common_name').size()
                         .reset_index(name='count')
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
                               STATUS_UNSURE: '#ff7f0e',
                           }),
                    use_container_width=True)

        st.download_button(
            '⬇️ Export filtered results as CSV',
            data=view_df.to_csv(index=False).encode('utf-8'),
            file_name=f'pipeline_{sel_batch_r}.csv', mime='text/csv')