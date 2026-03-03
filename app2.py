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

st.set_page_config(page_title="BirdNET UK/Ireland Analyzer", layout="wide")
st.title("🦆 BirdNET Audio Analyzer: UK & Ireland")

@st.cache_resource
def load_analyzer():
    return Analyzer()

LAT = 54.0
LON = -4.5

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header("Analysis Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)
st.sidebar.caption("Lower = more detections but more false positives.")

st.sidebar.markdown("---")
st.sidebar.header("Support")
st.sidebar.markdown(
    "For help or queries contact:\n\n"
    "**Mike Shewring**\n\n"
    "[mike.shewring@rspb.org.uk](mailto:mike.shewring@rspb.org.uk)"
)

# =============================================================================
# FILE UPLOADER
# =============================================================================

MAX_FILE_MB = 50
uploaded_files = st.file_uploader(
    "Upload Audio (WAV/MP3)",
    accept_multiple_files=True,
    type=["wav", "mp3"]
)

# =============================================================================
# HELPER: ANNOTATED SONOGRAM
# Draws a mel spectrogram and overlays coloured bands only for the
# species/detections the user has chosen to inspect.
# =============================================================================

def plot_annotated_sonogram(y, sr, detections, filename):
    fig, ax = plt.subplots(figsize=(14, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, x_axis="time", y_axis="mel", sr=sr, ax=ax, cmap="magma")

    species_list = list(set(d["common_name"] for d in detections)) if detections else []
    colour_map = {
        sp: plt.cm.tab20(i / max(len(species_list), 1))
        for i, sp in enumerate(species_list)
    }

    for d in detections:
        t_start = d["start_time"]
        t_end   = d["end_time"]
        colour  = colour_map[d["common_name"]]
        ax.axvspan(t_start, t_end, alpha=0.35, color=colour)
        ax.text(
            t_start + (t_end - t_start) / 2,
            ax.get_ylim()[1] * 0.92,
            f"{d['common_name']}\n{d['confidence']:.0%}",
            ha="center", va="top",
            fontsize=7, color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=colour, alpha=0.7)
        )

    if species_list:
        patches = [mpatches.Patch(color=colour_map[sp], label=sp) for sp in sorted(species_list)]
        ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.6,
                  ncol=max(1, len(species_list) // 6))

    ax.set_title(f"Annotated Sonogram: {filename}", fontsize=10, pad=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (mel)")
    fig.tight_layout()
    return fig

# =============================================================================
# MAIN APP LOGIC
# =============================================================================

if uploaded_files:

    valid_files = []
    for f in uploaded_files:
        if f.size > MAX_FILE_MB * 1024 * 1024:
            st.warning(f"'{f.name}' exceeds {MAX_FILE_MB}MB and will be skipped.")
        else:
            valid_files.append(f)

    if not valid_files:
        st.error("No valid files to process.")
        st.stop()

    # -------------------------------------------------------------------------
    # FILE PREVIEW
    # -------------------------------------------------------------------------
    st.subheader("File Preview")
    file_names = [f.name for f in valid_files]
    selected_preview = st.selectbox("Select a file to preview:", file_names)
    preview_file = next(f for f in valid_files if f.name == selected_preview)

    col1, col2 = st.columns([1, 3])
    with col1:
        preview_file.seek(0)
        st.audio(preview_file)
        st.caption("Sonogram preview capped at 60s for speed.")
    with col2:
        try:
            preview_file.seek(0)
            y_prev, sr_prev = librosa.load(preview_file, sr=None, duration=60)
            fig_prev, ax_prev = plt.subplots(figsize=(14, 3))
            S = librosa.feature.melspectrogram(y=y_prev, sr=sr_prev, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_db, x_axis="time", y_axis="mel", sr=sr_prev, ax=ax_prev, cmap="magma")
            ax_prev.set_title(f"Sonogram: {selected_preview}", fontsize=10)
            ax_prev.set_xlabel("Time (s)")
            ax_prev.set_ylabel("Frequency (mel)")
            fig_prev.tight_layout()
            st.pyplot(fig_prev)
            plt.close(fig_prev)
        except Exception as e:
            st.warning(f"Could not generate sonogram: {e}")

    # -------------------------------------------------------------------------
    # ANALYSIS BUTTON
    # -------------------------------------------------------------------------
    if st.button("Run BirdNET Analysis"):
        all_detections = []
        audio_cache    = {}
        analyzer       = load_analyzer()
        progress       = st.progress(0)
        status         = st.empty()

        for i, uploaded_file in enumerate(valid_files):
            status.info(f"Analysing {uploaded_file.name}  ({i+1}/{len(valid_files)})")
            suffix   = os.path.splitext(uploaded_file.name)[1]
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                recording = Recording(analyzer, tmp_path, lat=LAT, lon=LON, min_conf=conf_threshold)
                recording.analyze()

                uploaded_file.seek(0)
                y_audio, sr_audio = librosa.load(uploaded_file, sr=None, duration=120)
                audio_cache[uploaded_file.name] = (y_audio, sr_audio)

                if recording.detections:
                    for d in recording.detections:
                        d["file_name"] = uploaded_file.name
                        all_detections.append(d)
                else:
                    st.info(f"No detections above threshold in {uploaded_file.name}")

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

            progress.progress((i + 1) / len(valid_files))

        status.empty()

        # Store results in session_state so they persist when the user
        # interacts with the sonogram explorer below without re-running analysis
        st.session_state["all_detections"] = all_detections
        st.session_state["audio_cache"]    = audio_cache
        st.session_state["valid_file_names"] = [f.name for f in valid_files]

# =============================================================================
# RESULTS — drawn from session_state so they survive widget interactions
# =============================================================================

if "all_detections" in st.session_state and st.session_state["all_detections"]:
    all_detections   = st.session_state["all_detections"]
    audio_cache      = st.session_state["audio_cache"]
    valid_file_names = st.session_state["valid_file_names"]

    df = pd.DataFrame(all_detections)
    st.success(f"Analysis complete: {len(df)} detections across {len(valid_file_names)} file(s).")

    # -------------------------------------------------------------------------
    # SUMMARY METRIC CARDS
    # -------------------------------------------------------------------------
    st.subheader("Top Species by Confidence")
    top_species = (
        df.groupby("common_name")["confidence"]
        .max().sort_values(ascending=False).head(5)
    )
    card_cols = st.columns(min(5, len(top_species)))
    for col, (species, conf) in zip(card_cols, top_species.items()):
        with col:
            count = len(df[df["common_name"] == species])
            st.metric(label=species, value=f"{conf:.0%}", delta=f"{count} detection(s)")

    # -------------------------------------------------------------------------
    # SONOGRAM EXPLORER
    # User picks a file and one or more species to overlay, keeping the
    # sonogram readable even when there are many detections.
    # -------------------------------------------------------------------------
    st.subheader("Sonogram Explorer")
    st.caption("Select a file and the species you want to inspect. Only chosen species are overlaid.")

    col_a, col_b = st.columns([1, 2])

    with col_a:
        # File picker — only show files that are in the audio cache
        sono_file = st.selectbox(
            "Select recording:",
            options=list(audio_cache.keys()),
            key="sono_file"
        )

    with col_b:
        # Species picker — only show species detected in the chosen file
        file_species = sorted(
            df[df["file_name"] == sono_file]["common_name"].unique()
        )
        if file_species:
            selected_species = st.multiselect(
                "Select species to overlay:",
                options=file_species,
                default=file_species[:3] if len(file_species) >= 3 else file_species,
                key="sono_species"
            )
        else:
            selected_species = []
            st.info("No detections in this file.")

    # Confidence filter so user can focus on high-quality detections only
    min_conf_sono = st.slider(
        "Minimum confidence to show on sonogram:",
        min_value=0.1, max_value=1.0,
        value=conf_threshold,
        step=0.05,
        key="sono_conf"
    )

    if sono_file in audio_cache and selected_species:
        y_sono, sr_sono = audio_cache[sono_file]

        # Filter detections to only what the user has selected
        visible_detections = [
            d for d in all_detections
            if d["file_name"] == sono_file
            and d["common_name"] in selected_species
            and d["confidence"] >= min_conf_sono
        ]

        if visible_detections:
            fig_sono = plot_annotated_sonogram(y_sono, sr_sono, visible_detections, sono_file)
            st.pyplot(fig_sono)
            plt.close(fig_sono)
            st.caption(f"Showing {len(visible_detections)} detection(s) for {len(selected_species)} species.")
        else:
            # Show unannotated sonogram if nothing passes the filters
            fig_blank, ax_blank = plt.subplots(figsize=(14, 4))
            S = librosa.feature.melspectrogram(y=y_sono, sr=sr_sono, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_db, x_axis="time", y_axis="mel", sr=sr_sono, ax=ax_blank, cmap="magma")
            ax_blank.set_title(f"Sonogram: {sono_file}", fontsize=10)
            ax_blank.set_xlabel("Time (s)")
            ax_blank.set_ylabel("Frequency (mel)")
            fig_blank.tight_layout()
            st.pyplot(fig_blank)
            plt.close(fig_blank)
            st.caption("No detections match current filters — showing unannotated sonogram.")

    # -------------------------------------------------------------------------
    # DETECTION TIMELINE
    # -------------------------------------------------------------------------
    st.subheader("Detection Timeline")
    st.caption("x-axis = seconds from start of recording. Bubble size = confidence.")
    timeline_fig = px.scatter(
        df,
        x="start_time",
        y="common_name",
        size="confidence",
        color="common_name",
        facet_col="file_name" if len(valid_file_names) > 1 else None,
        facet_col_wrap=2,
        hover_data={
            "file_name": True,
            "confidence": ":.2f",
            "start_time": ":.1f",
            "end_time": ":.1f",
            "common_name": False,
        },
        labels={
            "start_time": "Time from Recording Start (s)",
            "common_name": "Species",
        },
        title="Species Detections — Time from Recording Start",
        height=max(400, len(df["common_name"].unique()) * 45),
    )
    timeline_fig.update_layout(
        showlegend=False,
        yaxis={"categoryorder": "total ascending"},
        xaxis={"showgrid": True},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(timeline_fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # CONFIDENCE BOX PLOT
    # -------------------------------------------------------------------------
    st.subheader("Confidence Distribution by Species")
    box_fig = px.box(
        df, x="confidence", y="common_name", color="common_name",
        labels={"confidence": "Confidence", "common_name": "Species"},
        height=max(400, len(df["common_name"].unique()) * 35),
    )
    box_fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(box_fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # FILTERED TABLE + CSV DOWNLOAD
    # -------------------------------------------------------------------------
    st.subheader("Raw Detections")
    species_filter = st.multiselect(
        "Filter by species:",
        options=sorted(df["common_name"].unique()),
        default=sorted(df["common_name"].unique()),
    )
    file_filter = st.multiselect(
        "Filter by file:",
        options=sorted(df["file_name"].unique()),
        default=sorted(df["file_name"].unique()),
    ) if len(valid_file_names) > 1 else [valid_file_names[0]]

    filtered_df = df[
        df["common_name"].isin(species_filter) &
        df["file_name"].isin(file_filter)
    ]
    st.dataframe(
        filtered_df.style.background_gradient(subset=["confidence"], cmap="RdYlGn"),
        use_container_width=True,
    )

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_filtered = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Filtered Detections as CSV",
            data=csv_filtered,
            file_name="birdnet_detections_filtered.csv",
            mime="text/csv",
        )
    with col_dl2:
        csv_all = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download All Detections as CSV",
            data=csv_all,
            file_name="birdnet_detections_all.csv",
            mime="text/csv",
        )

elif "all_detections" in st.session_state and not st.session_state["all_detections"]:
    st.warning("No detections found. Try lowering the confidence threshold.")


