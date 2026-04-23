import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NPA Impact on Bank Profitability",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 NPA Impact on Bank Profitability (ROA Predictor)")
st.markdown("**Problem Statement:** Predict bank profitability (ROA) using NPAs and key financial indicators.")
st.markdown("---")

# ─────────────────────────────────────────────
# Sidebar – Upload & Settings
# ─────────────────────────────────────────────
st.sidebar.header("📂 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Indian_Banks.csv", type=["csv"])

st.sidebar.header("⚙️ Model Settings")
test_size = st.sidebar.slider("Test Split Size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", value=42, step=1)
poly_degree = st.sidebar.selectbox("Polynomial Degree", [2, 3], index=0)
rf_estimators = st.sidebar.slider("Random Forest Trees", 50, 300, 100, 50)

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding='latin1')
    return df

if uploaded_file:
    df_raw = load_data(uploaded_file)
else:
    st.info("👆 Upload **Indian_Banks.csv** in the sidebar to get started. Using a demo placeholder until then.")
    st.stop()

# ─────────────────────────────────────────────
# Tab Layout
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview",
    "📈 EDA & Visualizations",
    "🤖 Model Training",
    "📉 Model Comparison",
    "🔮 Predict ROA"
])

# ══════════════════════════════════════════════
# TAB 1 – DATA OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Raw Data")
    st.dataframe(df_raw, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df_raw.shape[0])
    col2.metric("Columns", df_raw.shape[1])
    col3.metric("Missing Values", int(df_raw.isnull().sum().sum()))

    st.subheader("Missing Values (% per column)")
    missing_pct = (df_raw.isnull().sum() / len(df_raw) * 100).reset_index()
    missing_pct.columns = ["Column", "Missing %"]
    st.dataframe(missing_pct, use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df_raw.describe(), use_container_width=True)


# ──────────────────────────────────────────────
# Preprocessing (shared across tabs)
# ──────────────────────────────────────────────
df = df_raw.copy()
df = df.fillna(df.mean(numeric_only=True))

# Rename columns for convenience
col_map = {
    'ROA (%)\\n[DV]'             : 'ROA',
    'Net NPA\\nRatio (%)'         : 'Net_NPA_Ratio',
    'CAR / CRAR\\n(%)'            : 'CAR',
    'Credit Growth\\n(%)'         : 'Credit_Growth',
    'Cost-to-\\nIncome\\nRatio (%)': 'Cost_to_Income',
    'Bank Size\\n[Log(Assets)]'   : 'Log_Assets',
    'NPA × CAR\\n[Interaction]'   : 'NPA_CAR',
    'NPA × Size\\n[Interaction]'  : 'NPA_Size',
}
df.rename(columns=col_map, inplace=True)

# Recreate interaction features (ensure they exist)
df['NPA_CAR']  = df['Net_NPA_Ratio'] * df['CAR']
df['NPA_Size'] = df['Net_NPA_Ratio'] * df['Log_Assets']

# Outlier removal (IQR)
numeric_cols = df.select_dtypes(include=np.number).columns
Q1  = df[numeric_cols].quantile(0.25)
Q3  = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
mask = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
df  = df[mask].reset_index(drop=True)

FEATURES = ['Net_NPA_Ratio', 'CAR', 'Log_Assets', 'Cost_to_Income', 'Credit_Growth', 'NPA_CAR', 'NPA_Size']
TARGET   = 'ROA'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))


# ══════════════════════════════════════════════
# TAB 2 – EDA
# ══════════════════════════════════════════════
with tab2:
    st.subheader("ROA Distribution")
    fig, ax = plt.subplots()
    ax.hist(df[TARGET], bins=20, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("ROA (%)"); ax.set_ylabel("Frequency"); ax.set_title("ROA Distribution")
    st.pyplot(fig); plt.close()

    st.subheader("Scatter Plots: Features vs ROA")
    scatter_cols = st.multiselect(
        "Choose features to plot against ROA",
        FEATURES,
        default=['Net_NPA_Ratio', 'CAR', 'Cost_to_Income']
    )
    if scatter_cols:
        cols_per_row = 3
        rows = (len(scatter_cols) + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 4 * rows))
        axes = np.array(axes).flatten()
        for i, feat in enumerate(scatter_cols):
            axes[i].scatter(df[feat], df[TARGET], alpha=0.5, color="#DD8452")
            axes[i].set_xlabel(feat); axes[i].set_ylabel("ROA"); axes[i].set_title(f"{feat} vs ROA")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 7))
    corr = df[FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════
# TAB 3 – MODEL TRAINING
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Model Training Results")
    st.markdown(f"Training on **{len(X_train)}** samples | Testing on **{len(X_test)}** samples")

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_r2  = r2_score(y_test, y_pred_lr)
    lr_mse = mean_squared_error(y_test, y_pred_lr)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=rf_estimators, random_state=int(random_state))
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_r2  = r2_score(y_test, y_pred_rf)
    rf_mse = mean_squared_error(y_test, y_pred_rf)

    # Polynomial Regression
    poly = PolynomialFeatures(degree=poly_degree)
    X_poly       = poly.fit_transform(X)
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X_poly, y, test_size=test_size, random_state=int(random_state))
    model_poly = LinearRegression()
    model_poly.fit(X_train_p, y_train_p)
    y_pred_poly = model_poly.predict(X_test_p)
    poly_r2  = r2_score(y_test_p, y_pred_poly)
    poly_mse = mean_squared_error(y_test_p, y_pred_poly)

    # Metrics display
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 📐 Linear Regression")
        st.metric("R² Score", f"{lr_r2:.4f}")
        st.metric("MSE",      f"{lr_mse:.4f}")
    with c2:
        st.markdown("#### 🌲 Random Forest")
        st.metric("R² Score", f"{rf_r2:.4f}")
        st.metric("MSE",      f"{rf_mse:.4f}")
    with c3:
        st.markdown(f"#### 📐 Polynomial (deg={poly_degree})")
        st.metric("R² Score", f"{poly_r2:.4f}")
        st.metric("MSE",      f"{poly_mse:.4f}")

    # Actual vs Predicted plots
    st.subheader("Actual vs Predicted ROA")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    pairs = [
        (y_test,   y_pred_lr,   "Linear Regression"),
        (y_test,   y_pred_rf,   "Random Forest"),
        (y_test_p, y_pred_poly, f"Polynomial (deg={poly_degree})"),
    ]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for ax, (yt, yp, title), c in zip(axes, pairs, colors):
        ax.scatter(yt, yp, alpha=0.6, color=c)
        mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=1)
        ax.set_xlabel("Actual ROA"); ax.set_ylabel("Predicted ROA")
        ax.set_title(title)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Feature Importance
    st.subheader("🌲 Random Forest – Feature Importance")
    importance = rf.feature_importances_
    feat_df = pd.DataFrame({"Feature": FEATURES, "Importance": importance}).sort_values("Importance", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(feat_df["Feature"], feat_df["Importance"], color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("Importance"); ax.set_title("Feature Importance – Random Forest")
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    st.dataframe(feat_df.reset_index(drop=True), use_container_width=True)

    # Store models in session state
    st.session_state["lr"]         = lr
    st.session_state["rf"]         = rf
    st.session_state["model_poly"] = model_poly
    st.session_state["poly"]       = poly
    st.session_state["y_test"]     = y_test
    st.session_state["y_pred_lr"]  = y_pred_lr
    st.session_state["y_pred_rf"]  = y_pred_rf
    st.session_state["lr_r2"]      = lr_r2
    st.session_state["rf_r2"]      = rf_r2
    st.session_state["poly_r2"]    = poly_r2


# ══════════════════════════════════════════════
# TAB 4 – MODEL COMPARISON
# ══════════════════════════════════════════════
with tab4:
    st.subheader("Model Comparison Summary")
    comparison_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", f"Polynomial (deg={poly_degree})"],
        "R² Score": [lr_r2, rf_r2, poly_r2],
        "MSE":      [lr_mse, rf_mse, poly_mse],
    })
    st.dataframe(comparison_df.style.highlight_max(subset=["R² Score"], color="#b6e8b6")
                                    .highlight_min(subset=["MSE"],      color="#b6e8b6"),
                 use_container_width=True)

    best_model = comparison_df.loc[comparison_df["R² Score"].idxmax(), "Model"]
    st.success(f"✅ Best Model: **{best_model}** with R² = {comparison_df['R² Score'].max():.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(comparison_df["Model"], comparison_df["R² Score"],
                  color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("R² Score"); ax.set_title("R² Score Comparison")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, comparison_df["R² Score"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════
# TAB 5 – PREDICT ROA
# ══════════════════════════════════════════════
with tab5:
    st.subheader("🔮 Predict Bank ROA")
    st.markdown("Enter the financial indicators below to get an ROA prediction.")

    col_a, col_b = st.columns(2)
    with col_a:
        npa     = st.number_input("Net NPA Ratio (%)",        value=float(df['Net_NPA_Ratio'].mean()), step=0.1)
        car     = st.number_input("CAR / CRAR (%)",           value=float(df['CAR'].mean()),           step=0.1)
        log_a   = st.number_input("Bank Size [Log(Assets)]",  value=float(df['Log_Assets'].mean()),    step=0.1)
    with col_b:
        cti     = st.number_input("Cost-to-Income Ratio (%)", value=float(df['Cost_to_Income'].mean()),step=0.1)
        cg      = st.number_input("Credit Growth (%)",        value=float(df['Credit_Growth'].mean()), step=0.1)
        model_choice = st.selectbox("Choose Model for Prediction",
                                    ["Linear Regression", "Random Forest", f"Polynomial (deg={poly_degree})"])

    npa_car  = npa * car
    npa_size = npa * log_a
    input_arr = np.array([[npa, car, log_a, cti, cg, npa_car, npa_size]])

    if st.button("🔮 Predict ROA", use_container_width=True):
        if "lr" not in st.session_state:
            st.warning("Please go to 'Model Training' tab first to train the models.")
        else:
            lr_m     = st.session_state["lr"]
            rf_m     = st.session_state["rf"]
            poly_m   = st.session_state["model_poly"]
            poly_enc = st.session_state["poly"]

            if model_choice == "Linear Regression":
                pred = lr_m.predict(input_arr)[0]
            elif model_choice == "Random Forest":
                pred = rf_m.predict(input_arr)[0]
            else:
                pred = poly_m.predict(poly_enc.transform(input_arr))[0]

            st.success(f"**Predicted ROA: {pred:.4f} %**")
            st.markdown(f"*Model used: {model_choice}*")

            st.subheader("Input Summary")
            summary = pd.DataFrame({
                "Feature": FEATURES,
                "Value":   [npa, car, log_a, cti, cg, npa_car, npa_size]
            })
            st.dataframe(summary, use_container_width=True)
