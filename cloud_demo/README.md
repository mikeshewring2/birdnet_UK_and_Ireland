# Web demo build

A reduced, BirdNET-only build of the Audio Analyzer for deployment on
Streamlit Community Cloud. Intended for demonstration to funders and partners.

**`app2.py` in the repository root is the full desktop version and is not
affected by anything in this directory.** Users running it on their own PCs
continue to install from the root `requirements.txt` as before.

## Repository layout

```
your_repository/
├── .streamlit/
│   └── config.toml           # must be at the root — Community Cloud reads only this one
├── app2.py                   # full desktop version — UNCHANGED
├── requirements.txt          # full desktop dependencies — UNCHANGED
└── cloud_demo/
    ├── streamlit_app.py      # web demo entrypoint
    ├── requirements.txt      # slim, BirdNET-only — takes precedence over the root file
    ├── packages.txt          # apt packages (ffmpeg, libsndfile1)
    └── README.md             # this file
```

Community Cloud searches the entrypoint's directory for a dependency file
before falling back to the repository root, so `cloud_demo/requirements.txt`
is the one that gets installed. The root file is ignored for this app.

## Deploying

1. Push the repository to GitHub.
2. On share.streamlit.io, choose **New app** → your repo → branch.
3. Set **Main file path** to `cloud_demo/streamlit_app.py`.
4. Open **Advanced settings** and set **Python version** to **3.11**.
   This is not optional — `tflite-runtime` publishes no wheels for 3.12 or 3.13,
   and the previous deployment's segfault traces back in part to running on 3.13.
5. Deploy.

If you are changing the Python version of an existing app, delete the app and
redeploy rather than editing it in place.

## Why this build excludes Perch

`bioacoustics-model-zoo` depends on OpenSoundscape, which depends on PyTorch.
The desktop app imports that alongside TensorFlow. Both ship their own OpenMP
runtime, and loading both into a single process on Community Cloud's flat
PyPI-wheel environment causes a segmentation fault during import.

That crash happens in the C loader, so the `try/except ImportError` fallback in
`app2.py` cannot catch it — the process dies before Python regains control.
This is why the app failed on the server while working fine in a local conda
environment, where torch and TensorFlow share a single OpenMP library.

Removing Perch also keeps the deployment inside Community Cloud's memory and
disk limits: TensorFlow is roughly 600 MB and PyTorch pulls in CUDA wheels on
Linux even when no GPU is present.

## Demo limits

Set at the top of `streamlit_app.py`:

| Setting | Value | Reason |
|---|---|---|
| `MAX_FILE_MB` | 25 | Container memory |
| `MAX_ANALYSIS_SEC` | 300 | Keeps analysis responsive during a live demo |
| `MAX_BATCH_FILES` | 10 | Avoids long-running requests timing out |

## Known constraint: results are not persistent

The demo writes to SQLite in the container's temp directory. Community Cloud
wipes the filesystem on every reboot and redeploy, so batch results are lost
when the app restarts. The UI states this. If the demo ever needs to retain
data between sessions, point it at a hosted Postgres instance via
`st.connection` rather than a local file.

## Running the demo locally

Run from the repository root so paths resolve as they do on Community Cloud:

```bash
streamlit run cloud_demo/streamlit_app.py
```
