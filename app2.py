import streamlit as st
import os
import tempfile
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime, date
import io
import soundfile as sf

st.set_page_config(page_title='BirdNET Analyzer', layout='wide')
st.title('BirdNET Audio Analyzer')

@st.cache_resource
def load_analyzer():
    return Analyzer()

st.sidebar.header('Analysis Settings')

st.sidebar.subheader('Location')
lat = st.sidebar.number_input('Latitude',  value=54.0, format='%.4f')
lon = st.sidebar.number_input('Longitude', value=-4.5, format='%.4f')
st.sidebar.caption('Default: central UK/Ireland. Change for other regions.')

st.sidebar.subheader('Recording Date (optional)')
use_date = st.sidebar.checkbox('Apply seasonal date filter', value=False)
if use_date:
    rec_date = st.sidebar.date_input('Recording date', value=date.today())
    rec_time = st.sidebar.time_input('Recording time', value=datetime.now().time())
    st.sidebar.caption('BirdNET uses date and time to filter species by season.')
else:
    rec_date = None
    rec_time = None
    st.sidebar.caption('No date filter — all species considered.')

st.sidebar.subheader('Detection Settings')
conf_threshold = st.sidebar.slider('Confidence Threshold', 0.1, 1.0, 0.25)
st.sidebar.caption('Lower = more detections but more false positives.')
st.sidebar.markdown('---')
st.sidebar.header('Support')
st.sidebar.markdown('For help contact:\n\n**Mike Shewring**\n\n[mike.shewring@rspb.org.uk](mailto:mike.shewring@rspb.org.uk)')

MAX_FILE_MB = 500
uploaded_files = st.file_uploader('Upload Audio (WAV/MP3)', accept_multiple_files=True, type=['wav','mp3'])

def plot_annotated_sonogram(y, sr, detections, filename, t_start_sec=0, t_end_sec=60):
    s_start = int(t_start_sec * sr)
    s_end   = int(t_end_sec   * sr)
    y_slice = y[s_start:min(s_end, len(y))]
    fig, ax = plt.subplots(figsize=(14, 4))
    S    = librosa.feature.melspectrogram(y=y_slice, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, ax=ax, cmap='magma')
    window_detections = [d for d in detections if d['end_time'] >= t_start_sec and d['start_time'] <= t_end_sec]
    species_list = list(set(d['common_name'] for d in window_detections))
    colour_map = {sp: plt.cm.tab20(i / max(len(species_list),1)) for i,sp in enumerate(species_list)}
    for d in window_detections:
        rel_start = max(d['start_time'] - t_start_sec, 0)
        rel_end   = min(d['end_time']   - t_start_sec, t_end_sec - t_start_sec)
        colour    = colour_map[d['common_name']]
        ax.axvspan(rel_start, rel_end, alpha=0.35, color=colour)
        ax.text(rel_start+(rel_end-rel_start)/2, ax.get_ylim()[1]*0.92,
                f"{d['common_name']}\n{d['confidence']:.0%}",
                ha='center', va='top', fontsize=7, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=colour, alpha=0.7))
    if species_list:
        patches = [mpatches.Patch(color=colour_map[sp], label=sp) for sp in sorted(species_list)]
        ax.legend(handles=patches, loc='upper right', fontsize=7, framealpha=0.6, ncol=max(1,len(species_list)//6))
    ax.set_title(f'{filename}  [{t_start_sec:.0f}s - {t_end_sec:.0f}s]', fontsize=10, pad=8)
    ax.set_xlabel('Time in window (s)')
    ax.set_ylabel('Frequency (mel)')
    fig.tight_layout()
    return fig

if uploaded_files:
    valid_files = []
    for f in uploaded_files:
        if f.size > MAX_FILE_MB * 1024 * 1024:
            st.warning(f"'{f.name}' exceeds {MAX_FILE_MB}MB and will be skipped.")
        else:
            valid_files.append(f)
    if not valid_files:
        st.error('No valid files to process.')
        st.stop()

    st.subheader('File Preview')
    file_names = [f.name for f in valid_files]
    selected_preview = st.selectbox('Select a file to preview:', file_names)
    preview_file = next(f for f in valid_files if f.name == selected_preview)
    col1, col2 = st.columns([1, 3])
    with col1:
        preview_file.seek(0)
        st.audio(preview_file)
    with col2:
        try:
            preview_file.seek(0)
            y_full, sr_full = librosa.load(preview_file, sr=None)
            total_duration  = librosa.get_duration(y=y_full, sr=sr_full)
            window_size = st.select_slider('Sonogram window size (s):', options=[30,60,120,300], value=60)
            max_start = max(0.0, total_duration - window_size)
            window_start = st.slider('Scroll through recording:', min_value=0.0,
                max_value=float(int(max_start)) if max_start>0 else 0.0,
                value=0.0, step=float(window_size//2), format='%.0fs') if max_start>0 else 0.0
            window_end = min(window_start + window_size, total_duration)
            s0 = int(window_start * sr_full)
            s1 = int(window_end   * sr_full)
            y_slice = y_full[s0:s1]
            fig_prev, ax_prev = plt.subplots(figsize=(14,3))
            S    = librosa.feature.melspectrogram(y=y_slice, sr=sr_full, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr_full, ax=ax_prev, cmap='magma')
            ax_prev.set_title(f'Sonogram: {selected_preview}  [{window_start:.0f}s-{window_end:.0f}s / {total_duration:.0f}s]', fontsize=10)
            ax_prev.set_xlabel('Time in window (s)')
            ax_prev.set_ylabel('Frequency (mel)')
            fig_prev.tight_layout()
            st.pyplot(fig_prev)
            plt.close(fig_prev)
            st.caption(f'Total recording duration: {total_duration:.1f}s')
        except Exception as e:
            st.warning(f'Could not generate sonogram: {e}')

    if st.button('Run BirdNET Analysis'):
        all_detections = []
        audio_cache    = {}
        analyzer       = load_analyzer()
        progress       = st.progress(0)
        status         = st.empty()
        for i, uploaded_file in enumerate(valid_files):
            status.info(f'Analysing {uploaded_file.name}  ({i+1}/{len(valid_files)})')
            suffix   = os.path.splitext(uploaded_file.name)[1]
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                rec_kwargs = dict(lat=lat, lon=lon, min_conf=conf_threshold)
                if use_date and rec_date:
                    rec_kwargs['date'] = datetime.combine(rec_date, rec_time)
                recording = Recording(analyzer, tmp_path, **rec_kwargs)
                recording.analyze()
                uploaded_file.seek(0)
                y_audio, sr_audio = librosa.load(uploaded_file, sr=None)
                audio_cache[uploaded_file.name] = (y_audio, sr_audio)
                if recording.detections:
                    for d in recording.detections:
                        d['file_name'] = uploaded_file.name
                        all_detections.append(d)
                else:
                    st.info(f'No detections above threshold in {uploaded_file.name}')
            except Exception as e:
                st.error(f'Error processing {uploaded_file.name}: {e}')
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            progress.progress((i+1)/len(valid_files))
        status.empty()
        st.session_state['all_detections']   = all_detections
        st.session_state['audio_cache']      = audio_cache
        st.session_state['valid_file_names'] = [f.name for f in valid_files]
        if all_detections:
            st.session_state['validation'] = ['Unreviewed'] * len(all_detections)

if 'all_detections' in st.session_state and st.session_state['all_detections']:
    all_detections   = st.session_state['all_detections']
    audio_cache      = st.session_state['audio_cache']
    valid_file_names = st.session_state['valid_file_names']
    validation       = st.session_state.get('validation', ['Unreviewed']*len(all_detections))
    df = pd.DataFrame(all_detections)
    df['validation'] = validation
    st.success(f'Analysis complete: {len(df)} detections across {len(valid_file_names)} file(s).')
    tab_results, tab_validation = st.tabs(['Results', 'Validation'])

    with tab_results:
        st.subheader('Top Species by Confidence')
        top_species = df.groupby('common_name')['confidence'].max().sort_values(ascending=False).head(5)
        card_cols = st.columns(min(5, len(top_species)))
        for col,(species,conf) in zip(card_cols, top_species.items()):
            with col:
                count = len(df[df['common_name']==species])
                st.metric(label=species, value=f'{conf:.0%}', delta=f'{count} detection(s)')

        st.subheader('Sonogram Explorer')
        st.caption('Select a file and species to overlay. Scroll through long recordings with the window slider.')
        col_a, col_b = st.columns([1,2])
        with col_a:
            sono_file = st.selectbox('Select recording:', options=list(audio_cache.keys()), key='sono_file')
        with col_b:
            file_species = sorted(df[df['file_name']==sono_file]['common_name'].unique())
            if file_species:
                selected_species = st.multiselect('Select species to overlay:',
                    options=file_species,
                    default=file_species[:3] if len(file_species)>=3 else file_species,
                    key='sono_species')
            else:
                selected_species = []
                st.info('No detections in this file.')
        min_conf_sono = st.slider('Minimum confidence to show:', min_value=0.1, max_value=1.0,
            value=conf_threshold, step=0.05, key='sono_conf')
        if sono_file in audio_cache:
            y_sono,sr_sono = audio_cache[sono_file]
            total_dur = librosa.get_duration(y=y_sono, sr=sr_sono)
            sono_window = st.select_slider('Sonogram window size (s):',
                options=[30,60,120,300], value=60, key='sono_window')
            max_start_sono = max(0.0, total_dur - sono_window)
            sono_start = st.slider('Scroll through recording:',
                min_value=0.0,
                max_value=float(int(max_start_sono)) if max_start_sono>0 else 0.0,
                value=0.0, step=float(sono_window//2), format='%.0fs', key='sono_scroll'
            ) if max_start_sono>0 else 0.0
            sono_end = min(sono_start + sono_window, total_dur)
            visible_detections = [d for d in all_detections
                if d['file_name']==sono_file
                and d['common_name'] in selected_species
                and d['confidence']>=min_conf_sono] if selected_species else []
            fig_sono = plot_annotated_sonogram(y_sono, sr_sono, visible_detections, sono_file,
                t_start_sec=sono_start, t_end_sec=sono_end)
            st.pyplot(fig_sono)
            plt.close(fig_sono)
            in_window = [d for d in visible_detections if d['end_time']>=sono_start and d['start_time']<=sono_end]
            st.caption(f'Showing {len(in_window)} detection(s) in window. Recording duration: {total_dur:.1f}s')

        st.subheader('Detection Timeline')
        timeline_fig = px.scatter(df, x='start_time', y='common_name',
            size='confidence', color='common_name',
            facet_col='file_name' if len(valid_file_names)>1 else None, facet_col_wrap=2,
            hover_data={'file_name':True,'confidence':':.2f','start_time':':.1f','end_time':':.1f','common_name':False},
            labels={'start_time':'Time from Recording Start (s)','common_name':'Species'},
            title='Species Detections - Time from Recording Start',
            height=max(400, len(df['common_name'].unique())*45))
        timeline_fig.update_layout(showlegend=False,
            yaxis={'categoryorder':'total ascending'}, xaxis={'showgrid':True},
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(timeline_fig, use_container_width=True)

        st.subheader('Confidence Distribution by Species')
        box_fig = px.box(df, x='confidence', y='common_name', color='common_name',
            labels={'confidence':'Confidence','common_name':'Species'},
            height=max(400, len(df['common_name'].unique())*35))
        box_fig.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(box_fig, use_container_width=True)

        st.subheader('Raw Detections')
        species_filter = st.multiselect('Filter by species:',
            options=sorted(df['common_name'].unique()), default=sorted(df['common_name'].unique()))
        file_filter = st.multiselect('Filter by file:',
            options=sorted(df['file_name'].unique()), default=sorted(df['file_name'].unique())
        ) if len(valid_file_names)>1 else [valid_file_names[0]]
        filtered_df = df[df['common_name'].isin(species_filter) & df['file_name'].isin(file_filter)]
        st.dataframe(filtered_df.style.background_gradient(subset=['confidence'], cmap='RdYlGn'), use_container_width=True)
        col_dl1,col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button('Download Filtered CSV',
                data=filtered_df.to_csv(index=False).encode('utf-8'),
                file_name='birdnet_detections_filtered.csv', mime='text/csv')
        with col_dl2:
            st.download_button('Download All Detections CSV',
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='birdnet_detections_all.csv', mime='text/csv')

    with tab_validation:
        st.subheader('Validate Detections')
        st.caption('Mark each detection as Accepted, Rejected or Unsure. Status is included in the CSV export.')
        val_col1, val_col2 = st.columns(2)
        with val_col1:
            val_species_filter = st.multiselect('Show species:',
                options=sorted(df['common_name'].unique()),
                default=sorted(df['common_name'].unique()), key='val_species')
        with val_col2:
            val_status_filter = st.multiselect('Show status:',
                options=['Unreviewed','Accepted','Rejected','Unsure'],
                default=['Unreviewed','Accepted','Rejected','Unsure'], key='val_status')

        val_counts = pd.Series(validation).value_counts()
        v1, v2, v3, v4 = st.columns(4)
        v1.metric('Total',      len(validation))
        v2.metric('Accepted',   val_counts.get('Accepted',   0))
        v3.metric('Rejected',   val_counts.get('Rejected',   0))
        v4.metric('Unreviewed', val_counts.get('Unreviewed', 0))
        st.markdown('---')

        for idx, (det, val_status) in enumerate(zip(all_detections, validation)):
            if det['common_name'] not in val_species_filter: continue
            if val_status not in val_status_filter: continue

            border = {
                'Accepted':   '#2ecc71',
                'Rejected':   '#e74c3c',
                'Unsure':     '#f39c12',
                'Unreviewed': '#888888'
            }.get(val_status, '#888888')

            st.markdown(
                f'<div style="border-left:4px solid {border};padding-left:10px;margin-bottom:4px">',
                unsafe_allow_html=True)

            c1, c2 = st.columns([1, 2])

            with c1:
                st.markdown(f"**{det['common_name']}**  `{det['confidence']:.0%}`")
                st.markdown(f"{det['start_time']:.1f}s – {det['end_time']:.1f}s  |  _{det['file_name']}_")
                st.markdown(f"Status: **{val_status}**")
                b1, b2, b3 = st.columns(3)
                if b1.button('✅ Accept', key=f'acc_{idx}'):
                    st.session_state['validation'][idx] = 'Accepted'
                    st.rerun()
                if b2.button('❌ Reject', key=f'rej_{idx}'):
                    st.session_state['validation'][idx] = 'Rejected'
                    st.rerun()
                if b3.button('❓ Unsure', key=f'uns_{idx}'):
                    st.session_state['validation'][idx] = 'Unsure'
                    st.rerun()

            with c2:
                fname = det['file_name']
                if fname in audio_cache:
                    y_val, sr_val = audio_cache[fname]
                    pad     = 1.0
                    t_start = max(det['start_time'] - pad, 0)
                    t_end   = min(det['end_time'] + pad, librosa.get_duration(y=y_val, sr=sr_val))
                    s0      = int(t_start * sr_val)
                    s1      = int(t_end   * sr_val)
                    y_snip  = y_val[s0:s1]

                    try:
                        audio_buffer = io.BytesIO()
                        sf.write(audio_buffer, y_snip, sr_val, format='WAV')
                        audio_buffer.seek(0)
                        st.audio(audio_buffer, format='audio/wav')
                    except Exception:
                        st.caption('Could not render audio.')

                    try:
                        fig_v, ax_v = plt.subplots(figsize=(8, 2))
                        S    = librosa.feature.melspectrogram(y=y_snip, sr=sr_val, n_mels=128)
                        S_db = librosa.power_to_db(S, ref=np.max)
                        librosa.display.specshow(S_db, x_axis='time', y_axis='mel',
                                                 sr=sr_val, ax=ax_v, cmap='magma')
                        ax_v.axvspan(pad, pad + (det['end_time'] - det['start_time']),
                                     alpha=0.3, color='white')
                        ax_v.set_title(
                            f"{det['common_name']}  {det['start_time']:.1f}s–{det['end_time']:.1f}s",
                            fontsize=8)
                        ax_v.set_xlabel('')
                        ax_v.set_ylabel('')
                        fig_v.tight_layout()
                        st.pyplot(fig_v)
                        plt.close(fig_v)
                    except Exception:
                        st.caption('Could not render spectrogram.')

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('---')
        df_val = df.copy()
        df_val['validation'] = validation
        st.download_button('Download Validated CSV',
            data=df_val.to_csv(index=False).encode('utf-8'),
            file_name='birdnet_detections_validated.csv', mime='text/csv')

elif 'all_detections' in st.session_state and not st.session_state['all_detections']:
    st.warning('No detections found. Try lowering the confidence threshold.')