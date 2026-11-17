# Handwriting Personality Analyzer — Cloud UI

This package adds a cloud-ready Streamlit interface around the supplied
`Handwriting_Personality_Analyzer_ML.py`.

## Backend preservation

The original supplied `.py` file is included **unchanged**.

`backend.py` is a UI-safe adapter containing the same computational pipeline:
- OpenCV grayscale conversion and preprocessing
- Gaussian blur + Otsu binary threshold
- 2x2 morphological opening
- text-line detection
- line-spacing, letter-spacing, slant, ink-darkness and stroke-width features
- the same weak-label thresholds
- 4,000 synthetic samples with 12% label noise
- StandardScaler
- Random Forest (200 trees, max_depth=8)
- Logistic Regression (max_iter=1000)
- per-trait accuracy comparison and selection
- K-Means with 4 clusters
- 2-component PCA
- the same personality descriptions and cluster blurbs
- the same radar normalization ranges

No new prediction model or personality logic has been introduced.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload:
   - `app.py`
   - `backend.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `Handwriting_Personality_Analyzer_ML.py`
3. On Streamlit Community Cloud, select the repository.
4. Set the main file to `app.py`.
5. Deploy.

The app needs no database, no API key, and no purchased hardware.

## User flow

Upload image → preprocessing → feature extraction → ML prediction →
K-Means/PCA archetype → report → charts.

## Important note

The supplied project itself states that the handwriting-personality premise
is folklore rather than science. The cloud UI preserves that disclaimer.
