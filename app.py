"""
app.py  –  Streamlit frontend for the Fracture Detection API
Run with:
    streamlit run app.py

Make sure the FastAPI backend (api.py) is running first:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import io
import requests
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
API_BASE_URL   = "http://localhost:8000"
PREDICT_URL    = f"{API_BASE_URL}/predict"
HEALTH_URL     = f"{API_BASE_URL}/health"

BODY_PART_CLASSES = ["Elbow", "Hand", "Shoulder"]

PART_ICONS = {"Elbow": "💪", "Hand": "🖐", "Shoulder": "🦴"}

# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Card container */
    .result-card {
        background: #1a1d27;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 16px;
        border: 1px solid #2a2d3a;
    }

    /* Fracture badge – danger */
    .badge-fractured {
        background: linear-gradient(135deg, #ff4b4b, #c0392b);
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        padding: 10px 22px;
        border-radius: 8px;
        display: inline-block;
        letter-spacing: 0.5px;
        box-shadow: 0 0 18px rgba(255,75,75,0.45);
    }

    /* Normal badge – success */
    .badge-normal {
        background: linear-gradient(135deg, #00c853, #007e33);
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        padding: 10px 22px;
        border-radius: 8px;
        display: inline-block;
        letter-spacing: 0.5px;
        box-shadow: 0 0 18px rgba(0,200,83,0.35);
    }

    /* Metric tile */
    .metric-tile {
        background: #22263a;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        border: 1px solid #2e3347;
    }
    .metric-tile .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #7eb6ff;
    }
    .metric-tile .label {
        font-size: 0.78rem;
        color: #8888aa;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Header banner */
    .top-banner {
        background: linear-gradient(135deg, #1e2236 0%, #12151e 100%);
        border-radius: 14px;
        padding: 28px 36px;
        margin-bottom: 28px;
        border: 1px solid #2a2d3a;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #12151e; }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: #1a1d27;
        border: 2px dashed #3a3d54;
        border-radius: 12px;
        padding: 20px;
    }

    /* Divider */
    hr { border-color: #2a2d3a; }

    /* Ensure plotly charts always stretch to full column width */
    [data-testid="stPlotlyChart"] > div { width: 100% !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def check_api_health() -> tuple[bool, list[str]]:
    try:
        r = requests.get(HEALTH_URL, timeout=4)
        if r.status_code == 200:
            data = r.json()
            return True, data.get("models_loaded", [])
    except Exception:
        pass
    return False, []


def call_predict_api(image_bytes: bytes, filename: str) -> dict:
    resp = requests.post(
        PREDICT_URL,
        files={"file": (filename, image_bytes, "image/jpeg")},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def gauge_chart(value: float, title: str, color: str) -> go.Figure:
    """Small semicircular gauge for a 0-1 probability value."""
    fig = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = round(value * 100, 1),
        number = {"suffix": "%", "font": {"size": 28, "color": "#e0e0e0"}},
        title  = {"text": title, "font": {"size": 13, "color": "#8888aa"}},
        gauge  = {
            "axis": {"range": [0, 100], "tickcolor": "#555", "tickfont": {"color": "#666"},
                "nticks": 6,
                "tickwidth": 1,
            },
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "#22263a",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  50], "color": "#1a1d27"},
                {"range": [50, 100], "color": "#1e2030"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": round(value * 100, 1),
            },
        },
    ))
    fig.update_layout(
        autosize= True,
        height=200,
        margin={"t": 40, "b": 10, "l": 20, "r": 20},
        paper_bgcolor="#1a1d27",
        font={"color": "#e0e0e0"},
    )
    return fig


def bar_chart_parts(prob_dict: dict, predicted: str) -> go.Figure:
    """Horizontal bar chart of body-part probabilities."""
    parts  = list(prob_dict.keys())
    values = [v * 100 for v in prob_dict.values()]
    colors = ["#7eb6ff" if p == predicted else "#3a3d54" for p in parts]

    fig = go.Figure(go.Bar(
        x           = values,
        y           = parts,
        orientation = "h",
        marker_color= colors,
        text        = [f"{v:.1f}%" for v in values],
        textposition= "outside",
        textfont    = {"color": "#e0e0e0", "size": 12},
    ))
    fig.update_layout(
        autosize      = True,
        height      = 180,
        margin      = {"t": 10, "b": 10, "l": 10, "r": 60},
        paper_bgcolor="#1a1d27",
        plot_bgcolor= "#1a1d27",
        xaxis       = {"range": [0, 110], "showgrid": False, "visible": False},
        yaxis       = {"tickfont": {"color": "#e0e0e0", "size": 13}},
        showlegend  = False,
    )
    return fig


PLOTLY_CFG = {"displayModeBar": False, "responsive": True}


# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦴 Fracture Detection")
    st.markdown("---")

    # API health indicator
    api_ok, loaded_models = check_api_health()
    if api_ok:
        st.success("✅  API Online")
        with st.expander("Loaded models"):
            for m in loaded_models:
                st.markdown(f"- `{m}`")
    else:
        st.error("❌  API Offline")
        st.caption(f"Expected at `{API_BASE_URL}`")
        st.info("Start the API with:\n```\nuvicorn api:app --port 8000\n```")

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "Three-phase pipeline:\n\n"
        "**Phase 0** – AI/real image check *(Gemini)*\n\n"
        "**Phase 1** – Body part ID *(ResNet50)*\n"
        "*(Elbow / Hand / Shoulder)*\n\n"
        "**Phase 2** – Fracture detection *(ResNet50)*\n"
        "*(Fractured / Normal)*"
    )
    st.markdown("---")
    st.caption("Supported formats: JPEG · PNG · WEBP")


# ─────────────────────────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
  <h1 style="margin:0;font-size:2rem;">🦴 X-Ray Fracture Detection</h1>
  <p style="color:#8888aa;margin:6px 0 0;">
    Upload an X-ray image — the pipeline verifies authenticity,
    identifies the bone, and checks for fractures.
  </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your X-ray image here or click to browse",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="visible",
)

# ─────────────────────────────────────────────────────────────────
# Results layout
# ─────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    pil_image   = Image.open(io.BytesIO(image_bytes))

    # Layout: image left, results right
    col_img, col_results = st.columns([1, 1.5], gap="large")

    with col_img:
        st.markdown("#### Uploaded X-Ray")
        st.image(pil_image, use_container_width=True, caption=uploaded_file.name)

    with col_results:
        if not api_ok:
            st.error("Cannot reach the API. Please start the backend first.")
            st.stop()

        with st.spinner("Running inference …"):
            try:
                result = call_predict_api(image_bytes, uploaded_file.name)
            except requests.exceptions.HTTPError as e:
                try:
                    err    = e.response.json()
                    detail = err.get("detail", {})
                    if isinstance(detail, dict) and detail.get("error") == "AI-generated image detected":
                        st.error("🚫 AI-generated image detected — upload blocked.")
                        ca, cb = st.columns(2)
                        ca.metric("AI Confidence", f"{detail.get('confidence', 0)*100:.1f}%")
                        cb.markdown(
                            f"<div class='result-card' style='margin-top:8px;'>"
                            f"<span style='color:#8888aa;font-size:0.85rem;'>Reason</span><br>"
                            f"<span>{detail.get('reason', '—')}</span></div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error(f"API error: {detail or err}")
                except Exception:
                    st.error(f"API returned an error: {e.response.text}")
                st.stop()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        # ── Phase 0 card ──────────────────────────────────────────
        st.markdown("#### 🧠 Phase 0 — Image Authenticity")
        ai_flag   = result.get("is_ai_generated", False)
        ai_conf   = result.get("ai_confidence", 0.0)
        ai_reason = result.get("ai_reason", "")
        ai_color  = "#ff4b4b" if ai_flag else "#00c853"
        ai_badge  = "🚫 AI Generated" if ai_flag else "✅ Real Image"

        if ai_flag:
            st.warning("⚠ This image may be AI-generated. Results may be unreliable.")

        reason_html = (
            f"<div style='color:#aaa;font-size:0.82rem;margin-top:6px;'>{ai_reason}</div>"
            if ai_reason else ""
        )
        st.markdown(f"""
        <div class="result-card">
          <div style="font-size:1.15rem;font-weight:700;color:{ai_color};">{ai_badge}</div>
          <div style="color:#8888aa;font-size:0.85rem;margin-top:8px;">
            Confidence: <strong style="color:#7eb6ff;">{ai_conf*100:.1f}%</strong>
          </div>
          {reason_html}
        </div>""", unsafe_allow_html=True)

        # ── Phase 1 card ──────────────────────────────────────────
        st.markdown("#### Phase 1 — Body Part")
        part_icon = PART_ICONS.get(result["predicted_part"], "🦴")
        st.markdown(f"""
        <div class="result-card">
          <span style="font-size:2rem;">{part_icon}</span>
          <span style="font-size:1.5rem; font-weight:700; margin-left:10px;">
            {result["predicted_part"]}
          </span>
          <div style="color:#8888aa; font-size:0.85rem; margin-top:6px;">
            Confidence: <strong style="color:#7eb6ff;">
              {result["part_confidence"]*100:.1f}%
            </strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Phase 2 result ─────────────────────────────────────────
        st.markdown("#### Phase 2 — Fracture Detection")
        badge_class = "badge-fractured" if result["is_fractured"] else "badge-normal"
        st.markdown(f"""
        <div class="result-card">
          <div class="{badge_class}">{result["fracture_label"]}</div>
          <div style="color:#8888aa; font-size:0.85rem; margin-top:12px;">
            Fracture probability: <strong style="color:#7eb6ff;">
              {result["fracture_confidence"]*100:.1f}%
            </strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Charts row — full width, 3 equal columns ──────────────────
    st.markdown("---")
    st.markdown("#### Confidence Charts")

    ch1, ch2, ch3 = st.columns(3, gap="medium")

    with ch1:
        st.markdown(
            "<p style='text-align:center;color:#8888aa;font-size:0.82rem;"
            "text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0;'>"
            "AI Detection</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            gauge_chart(ai_conf, "AI Confidence", ai_color),
            use_container_width=True,
            config=PLOTLY_CFG,
        )

    with ch2:
        st.markdown(
            "<p style='text-align:center;color:#8888aa;font-size:0.82rem;"
            "text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0;'>"
            "Part Identification</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            gauge_chart(result["part_confidence"], "Part Confidence", "#7eb6ff"),
            use_container_width=True,
            config=PLOTLY_CFG,
        )

    with ch3:
        st.markdown(
            "<p style='text-align:center;color:#8888aa;font-size:0.82rem;"
            "text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0;'>"
            "Fracture Probability</p>",
            unsafe_allow_html=True,
        )
        frac_color = "#ff4b4b" if result["is_fractured"] else "#00c853"
        st.plotly_chart(
            gauge_chart(result["fracture_confidence"], "Fracture Probability", frac_color),
            use_container_width=True,
            config=PLOTLY_CFG,
        )

    # ── Body-part probability bar chart ───────────────────────────
    st.markdown("#### Body Part Probabilities")
    st.plotly_chart(
        bar_chart_parts(result["part_probabilities"], result["predicted_part"]),
        use_container_width=True,
        config=PLOTLY_CFG,
    )

    # ── Summary metrics strip ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Summary")
    m1, m2, m3, m4, m5 = st.columns(5)

    tiles = [
        (m1, result["predicted_part"],
             "Detected Part", "#7eb6ff"),
        (m2, f"{result['part_confidence']*100:.0f}%",
             "Part Confidence", "#7eb6ff"),
        (m3, "⚠ Yes" if result["is_fractured"] else "✓ No",
             "Fracture Present",
             "#ff4b4b" if result["is_fractured"] else "#00c853"),
        (m4, f"{result['fracture_confidence']*100:.0f}%",
             "Fracture Prob.", "#7eb6ff"),
        (m5, "AI" if result.get("is_ai_generated") else "Real",
             "Image Type",
             "#ff4b4b" if result.get("is_ai_generated") else "#00c853"),
    ]

    for col, val, lbl, clr in tiles:
        with col:
            st.markdown(f"""
            <div class="metric-tile">
              <div class="value" style="color:{clr};">{val}</div>
              <div class="label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # ── Raw JSON toggle ────────────────────────────────────────────
    with st.expander("🔍 Raw API response"):
        st.json(result)

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#555;">
      <div style="font-size:4rem;">🩻</div>
      <div style="font-size:1.1rem; margin-top:12px;">
        Upload an X-ray image above to begin
      </div>
      <div style="font-size:0.85rem; margin-top:6px; color:#444;">
        Supported body parts: Elbow · Hand · Shoulder
      </div>
    </div>
    """, unsafe_allow_html=True)