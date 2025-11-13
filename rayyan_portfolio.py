# streamlit_portfolio.py
# Data Scientist Portfolio Template (single-file Streamlit app)
# -------------------------------------------------------------
# HOW TO USE
# 1. Install dependencies: pip install streamlit pandas matplotlib
# 2. Run locally: streamlit run streamlit_portfolio.py
# 3. Replace placeholder text, images, project links and resume file
# 4. Deploy on Streamlit Community Cloud or Heroku/Render if desired

import streamlit as st
from PIL import Image
import base64
import io
import pandas as pd
import os
# ------------------------
# ---- CONFIG / DATA ----
# ------------------------
st.set_page_config(page_title="Syed Rayyan — Data Science ", page_icon=":bar_chart:", layout="wide")

# Replace these with your details
NAME = "Syed Rayyan"
TAGLINE = "Aspiring Data Scientist | Python · ML · Data Viz"
SHORT_BIO = (
    "I am a fresher in data science who enjoys turning data into actionable insights. "
    "I work with Python, SQL, and machine learning to solve real-world problems."
)
CONTACT_EMAIL = "sumamarayyan@gmail.com"
LINKEDIN = "https://www.linkedin.com/in/syedrayyan001"
GITHUB = "https://github.com/anonymousevil01"


# Sample projects - replace with your own
PROJECTS = [
    {
        "title": "Heart Disease Prediction",
        "subtitle": "Binary classification using clinical features",
        "description": (
            "Built and evaluated multiple classifiers (Logistic Regression, Random Forest, XGBoost). "
            "Performed feature engineering, class imbalance handling and model explainability."
        ),
        "tools": "Python, Pandas, Scikit-learn, SHAP",
        "github": "https://github.com/anonymousevil01/Heart-disease-Prediction",
       
        "image": "images/heart_disease.jpeg"


    },
    {
        "title": "FIFA Player Clustering",
        "subtitle": "Unsupervised clustering to segment player types",
        "description": (
            "Performed feature selection, dimensionality reduction (PCA), and KMeans clustering. "
            "Interpreted clusters and created visual dashboards."
        ),
        "tools": "Python, Scikit-learn, PCA, Plotly",
        "github": "https://github.com/anonymousevil01/FIFA-20-Player-Analysis",
        "image": "images/fifa player.jpeg",
    },
    {
        "title": "Game Winner Prediction",
        "subtitle": "Predict the final placement (win/top-10/rank) of a player or squad in a PUBG match using match statistics.",
        "description": (
            "Used machine learning models like Random Forest and XGBoost to analyze PUBG match data and predict the winner. These algorithms learn patterns from features like kills, damage, distance traveled, and survival time to estimate the player’s win probability. "
        "Dataset Used: PUBG match dataset (usually from Kaggle) containing player-level and match-level information."    
        ),
        "tools": "Python, Scikit-learn, PCA, Plotly, Random Forest,LightGBM, Linear Regression ",
        "github": "https://github.com/anonymousevil01/Game-Winner-Prediction",
        "image":"images/pubg.jpeg",
    },
    {
        "title": "The Future of Digital Security",
        "subtitle": "A Machine Learning Approach to Detecting Deepfake Content",
        "description": (
            "This project focuses on identifying manipulated or AI-generated videos using deep learning techniques. By analyzing facial movements, inconsistencies, and pixel-level artifacts, the model detects whether a video or image is real or fake. Convolutional Neural Networks (CNNs) and advanced architectures like EfficientNet or XceptionNet are used to classify content with high accuracy, helping prevent misinformation, fraud, and misuse of synthetic media. "
         
        ),
        "tools": "Python, Scikit-learn, MobileNet, Resnet,OpenCV, Numpy, pandas ",
        "github": "https://github.com/anonymousevil01/The-future-of-Digital-Security",
        "image": "images/deepfake.jpeg",
    },
    {
        "title": "Real-time Emotion Detection",
        "subtitle": "Detecting Facial Expressions in Real Time Using Deep Learning",
        "description": (
            "This project focuses on identifying human emotions such as happiness, sadness, anger, and surprise from live video or webcam feeds. Using computer vision and deep learning models like CNNs, the system detects faces in real time and classifies the corresponding emotion based on facial expressions. The model processes each frame instantly, enabling applications in human–computer interaction, mental health monitoring, gaming, and smart surveillance. "
         
        ),
        "tools": "Python, Scikit-learn,OpenCV, Numpy, pandas, haarcascade ",
        "github": "https://github.com/anonymousevil01/Real-Time-Emotion-Detection-",
        "image": "images/emotion.jpeg",
    },
]

# Optional resume bytes: replace by reading a real file if you have one
with open("resume.pdf", "rb") as f:
    RESUME_BYTES = f.read()

# ------------------------
# ---- PAGE LAYOUT -----
# ------------------------

# Header
col1, col2 = st.columns([1,2])
with col1:
    # Optionally show a profile image if provided
    try:
        profile = Image.open("PROFILE.jpg")  # put profile.jpg in same folder or change path
        st.image(profile, width=160)
    except Exception:
        st.write("")
with col2:
    st.title(NAME)
    st.write("#### " + TAGLINE)
    st.write(SHORT_BIO)

st.markdown("---")

# Skills and Links
left, middle, right = st.columns([1, 1, 1])
with left:
    st.header("Skills")
    st.write("**Programming:** Python, SQL")
    st.write("**Data Handling:** Pandas, NumPy")
    st.write("**ML & Tools:** Scikit-learn, XGBoost, TensorFlow (basics)")
    st.write("**Visualization:** Matplotlib, Seaborn, Plotly, Tableau")
    st.write("**Others:** Git, Excel, Model Explainability (SHAP)")

   
with middle:
    st.header("Certifications")
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.write("- Google Data Analytics Certificate")
    st.write("- IBM Data Science Professional Certificate")
    st.write("- Machine Learning with Python — Coursera")
    st.write("- Certified Data Scientist (Gold) | NASSCOM, 2025 ")
    st.write("- Data Scientist Certification Training | DataMites Global Training Institute ")
    st.markdown("</div>", unsafe_allow_html=True)
    

with right:
    st.header("9096837886")
    st.write(f"Email: {CONTACT_EMAIL}")
    st.write(f"[LinkedIn]({LINKEDIN})")
    st.write(f"[GitHub]({GITHUB})")
    
    st.download_button("Download Resume (PDF)", data=RESUME_BYTES, file_name="resume.pdf", mime="application/pdf")

st.markdown("---")

# Projects
st.header("Projects")

for proj in PROJECTS:
    st.subheader(proj["title"])
    st.write("_" + proj["subtitle"] + "_")

    desc_cols = st.columns([2, 1])
    with desc_cols[0]:
        st.write(proj["description"])
        st.write("**Tools:** " + proj["tools"])
        st.write(f"[GitHub]({proj['github']})")

    with desc_cols[1]:
        # IMAGE HANDLING: accept local path or http(s) url
        try:
            img_src = proj.get("image")
            if not img_src:
                raise FileNotFoundError("No image path set for project.")

            # Ensure image is a string (not a Streamlit display call mistakenly stored)
            if not isinstance(img_src, str):
                raise TypeError("Project image must be a path string or URL.")

            # Try local file first (handle space vs underscore by trying both)
            possible_paths = [img_src]
            # if filename contains space, also try underscore variant and vice-versa
            fname = os.path.basename(img_src)
            if " " in fname:
                possible_paths.append(os.path.join(os.path.dirname(img_src), fname.replace(" ", "_")))
            elif "_" in fname:
                possible_paths.append(os.path.join(os.path.dirname(img_src), fname.replace("_", " ")))

            opened = None
            for p in possible_paths:
                if os.path.exists(p):
                    opened = Image.open(p)
                    break

            # If not a local file, try URL
            if opened is None:
                if img_src.lower().startswith("http"):
                    resp = requests.get(img_src, timeout=10)
                    resp.raise_for_status()
                    opened = Image.open(io.BytesIO(resp.content))
                else:
                    # last resort: try to open the original path (may raise)
                    opened = Image.open(img_src)

            # Show image — use_container_width (new API)
            st.image(opened, caption=proj["title"], use_container_width=True)

        except Exception as e:
            # friendly fallback + optional debug info (remove debug in production)
            st.write("*(Add a screenshot or chart for this project)*")
            # optional: show small debug message for local dev
            st.caption(f"Image load error: {e}")
    st.markdown("---")
# Optional: Interactive demo section (replace with your code)
#st.header("Interactive Demos")
#st.write("Below you can add interactive widgets or small demos for your projects. For example, upload a CSV and show a preview and a chart.")

#with st.expander("Upload dataset and preview"):
#    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
 #   if uploaded_file is not None:
  #      df = pd.read_csv(uploaded_file)
   #     st.write(df.head())
    #    if st.checkbox("Show basic stats"):
     #       st.write(df.describe())

# Footer / Call to action
#st.markdown("---")
footer_col1, footer_col2 = st.columns([3, 1])
with footer_col1:
    st.write("If you'd like to see more projects or collaborate, feel free to reach out!")
with footer_col2:
    st.write(f"Email: {CONTACT_EMAIL}")

# ------------------------
# ---- END OF FILE ------
# ------------------------

# Tips / Next steps (edit these comments in your copy):
# - Replace placeholders (NAME, CONTACT_EMAIL, LINKS, PROJECTS)
# - Add real images/screenshots for each project (put them in the same folder and set proj['image'] to the filename)
# - Replace RESUME_BYTES by reading an actual PDF: open('resume.pdf','rb').read()
# - Add more interactive demos: model inference widgets, charts, and dashboards
# - To deploy: push to GitHub and connect to Streamlit Community Cloud (https://streamlit.io/cloud)