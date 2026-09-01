# -*- coding: utf-8 -*-
"""
H&M Big Data – RFM Clustering Dashboard
Streamlit app – Mục 4 Bài tập nhóm Big Data
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="H&M RFM Clustering Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root { color-scheme: light; }

/* Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1117;
    color: #f0f0f0;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Space Grotesk', sans-serif;
    color: #e8b4b8 !important;
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* KPI card */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e8ecf0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8a9bb0;
    margin-bottom: 0.35rem;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #1a1f2e;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.75rem;
    color: #aab5c2;
    margin-top: 0.25rem;
}

/* Cluster badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

/* Section header */
.section-eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #c06b72;
    margin-bottom: 0.3rem;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #1a1f2e;
    margin-bottom: 1.2rem;
    border-bottom: 2px solid #f0f0f4;
    padding-bottom: 0.5rem;
}

/* Prediction result box */
.pred-box {
    border-radius: 14px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin-top: 1rem;
}
.pred-cluster {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.pred-label {
    font-size: 1rem;
    font-weight: 500;
    opacity: 0.8;
}

/* Tab styling */
[data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: 2px solid #e8ecf0;
}
[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    border-radius: 8px 8px 0 0 !important;
    padding: 0.6rem 1.2rem !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CLUSTER METADATA (màu & nhãn)
# ──────────────────────────────────────────────────────────────────────────────

# Hardcode cho chế độ dữ liệu mặc định (K=5, H&M dataset)
CLUSTER_META_DEFAULT = {
    0: {"label": "New / Inactive",   "color": "#6d4c41", "bg": "#ede0d4", "tip": "Gửi ưu đãi kích hoạt & giảm giá đơn hàng tiếp theo"},
    1: {"label": "Loyal Customers",  "color": "#1565c0", "bg": "#dbeafe", "tip": "Cross-sell sản phẩm liên quan & thúc đẩy lên VIP"},
    2: {"label": "Super VIP",        "color": "#b8860b", "bg": "#fff8dc", "tip": "Chăm sóc 1-1, early access & quà tặng cá nhân hóa"},
    3: {"label": "Champions",        "color": "#2d6a4f", "bg": "#d8f3dc", "tip": "Giữ chân bằng loyalty reward & ambassador program"},
    4: {"label": "Potential Loyals", "color": "#6a0572", "bg": "#f3d9fa", "tip": "Khuyến khích lần mua tiếp theo & chương trình onboarding"},
}

# Bảng màu dynamic cho chế độ upload (tối đa 10 cluster)
_PALETTE = [
    {"color": "#6d4c41", "bg": "#ede0d4"},
    {"color": "#1565c0", "bg": "#dbeafe"},
    {"color": "#b8860b", "bg": "#fff8dc"},
    {"color": "#2d6a4f", "bg": "#d8f3dc"},
    {"color": "#6a0572", "bg": "#f3d9fa"},
    {"color": "#c62828", "bg": "#fce4ec"},
    {"color": "#00796b", "bg": "#e0f2f1"},
    {"color": "#4527a0", "bg": "#ede7f6"},
    {"color": "#558b2f", "bg": "#f1f8e9"},
    {"color": "#e65100", "bg": "#fff3e0"},
]

_TIPS = {
    "Champions":       "Giữ chân bằng loyalty reward & ambassador program",
    "Loyal Customers": "Cross-sell sản phẩm liên quan & thúc đẩy lên VIP",
    "Potential Loyals":"Khuyến khích lần mua tiếp theo & chương trình onboarding",
    "New / Inactive":  "Gửi ưu đãi kích hoạt & giảm giá đơn hàng tiếp theo",
    "At Risk":         "Win-back chi phí thấp, ưu tiên nếu historical value cao",
    "Super VIP":       "Chăm sóc 1-1, early access & quà tặng cá nhân hóa",
    "Can't Lose Them": "Cảnh báo sớm & offer cao cấp để giữ chân ngay lập tức",
    "Lost":            "Loại bỏ hoặc duy trì chi phí tối thiểu",
}

def _auto_label(cid, df_stats):
    """Tự động gán nhãn cluster dựa trên rank RFM và phát hiện outlier."""
    idx = df_stats[df_stats["cluster"] == cid].index[0]

    # Phát hiện Super VIP dùng IQR (robust, không bị outlier kéo ngưỡng)
    f_values = df_stats["Avg_Frequency"]
    f_val    = df_stats.loc[idx, "Avg_Frequency"]
    q1 = f_values.quantile(0.25)
    q3 = f_values.quantile(0.75)
    iqr = q3 - q1
    if f_val > q3 + 1.5 * iqr:
        return "Super VIP"

    # Loại Super VIP ra trước khi tính rank để tránh outlier kéo rank các cluster còn lại
    outlier_mask = df_stats["Avg_Frequency"] <= q3 + 1.5 * iqr
    df_rank = df_stats[outlier_mask]
    idx_rank = df_rank[df_rank["cluster"] == cid].index[0]

    # Rank tương đối (0–1), R đảo ngược vì R thấp = mua gần đây = tốt hơn
    r_rank = 1 - df_rank["Avg_Recency"].rank(pct=True)[idx_rank]
    f_rank = df_rank["Avg_Frequency"].rank(pct=True)[idx_rank]
    m_rank = df_rank["Avg_Monetary"].rank(pct=True)[idx_rank]
    score  = (r_rank * 0.3) + (f_rank * 0.4) + (m_rank * 0.3)

    # Can't Lose Them: F và M cao (từng là khách tốt) nhưng R thấp (lâu không mua)
    if r_rank <= 0.2 and f_rank >= 0.6 and m_rank >= 0.6:
        return "Can't Lose Them"

    # Lost: R, F, M đều rất thấp
    if r_rank <= 0.2 and f_rank <= 0.2 and m_rank <= 0.2:
        return "Lost"

    if score >= 0.8:   return "Champions"
    elif score >= 0.6: return "Loyal Customers"
    elif score >= 0.4: return "Potential Loyals"
    elif score >= 0.2: return "New / Inactive"
    else:              return "At Risk"

def _build_cluster_meta(df_stats):
    """Xây dựng CLUSTER_META động từ df_stats, xử lý trùng tên."""
    labels_raw = {int(row["cluster"]): _auto_label(int(row["cluster"]), df_stats)
                  for _, row in df_stats.iterrows()}
    counts = {}
    meta = {}
    for cid, label in sorted(labels_raw.items()):
        counts[label] = counts.get(label, 0) + 1
        final_label = f"{label} {counts[label]}" if counts[label] > 1 else label
        palette = _PALETTE[cid % len(_PALETTE)]
        meta[cid] = {
            "label": final_label,
            "color": palette["color"],
            "bg":    palette["bg"],
            "tip":   _TIPS.get(label, "Phân tích thêm để đưa ra chiến lược phù hợp"),
        }
    return meta

# CLUSTER_META sẽ được gán sau khi biết data_mode và load xong df_stats
CLUSTER_META = CLUSTER_META_DEFAULT

def cluster_meta(cid):
    return CLUSTER_META.get(int(cid), {"label": f"Cluster {cid}", "color": "#555", "bg": "#eee", "tip": ""})

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(seg_file, stats_file, ksel_file):
    df_seg   = pd.read_csv(seg_file)
    df_stats = pd.read_csv(stats_file)
    df_ksel  = pd.read_csv(ksel_file) if ksel_file is not None else None
    return df_seg, df_stats, df_ksel

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR – file upload
# ──────────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SEG   = os.path.join(_HERE, "data", "customer_segments.csv")
DEFAULT_STATS = os.path.join(_HERE, "data", "cluster_rfm_stats.csv")
DEFAULT_KSEL  = os.path.join(_HERE, "data", "k_means_k_selection_summary.csv")

with st.sidebar:
    st.markdown("## 🛍️ H&M RFM Dashboard")
    st.markdown("---")

    data_mode = st.radio(
        "Nguồn dữ liệu",
        ["📂 Dùng dữ liệu mặc định", "⬆️ Upload file của bạn"],
        index=0,
    )

    f_seg = f_stats = f_ksel = None

    if data_mode == "⬆️ Upload file của bạn":
        st.markdown("### Tải dữ liệu lên")
        f_seg   = st.file_uploader("customer_segments.csv",     type="csv", key="seg")
        f_stats = st.file_uploader("cluster_rfm_stats.csv",     type="csv", key="stats")
        f_ksel  = st.file_uploader("k_means_k_selection_summary.csv (tuỳ chọn)", type="csv", key="ksel")

    st.markdown("---")
    st.caption("Bài tập nhóm – Big Data | H&M Dataset")

# Xác định nguồn file thực sự sẽ dùng
if data_mode == "📂 Dùng dữ liệu mặc định":
    if not os.path.exists(DEFAULT_SEG) or not os.path.exists(DEFAULT_STATS):
        st.error("⚠️ Không tìm thấy file mặc định trong thư mục `data/`. Vui lòng chọn **Upload file của bạn** hoặc thêm file vào repo.")
        st.stop()
    seg_source   = DEFAULT_SEG
    stats_source = DEFAULT_STATS
    ksel_source  = DEFAULT_KSEL if os.path.exists(DEFAULT_KSEL) else None
else:
    if f_seg is None or f_stats is None:
        st.markdown("""
        <div style='text-align:center; padding: 4rem 2rem;'>
            <div style='font-size:3.5rem; margin-bottom:1rem;'>🛍️</div>
            <div style='font-family: Space Grotesk, sans-serif; font-size:1.6rem; font-weight:700; color:#1a1f2e; margin-bottom:0.5rem;'>
                H&M RFM Clustering Dashboard
            </div>
            <div style='color:#8a9bb0; font-size:0.95rem; max-width:420px; margin:0 auto;'>
                Tải lên <strong>customer_segments.csv</strong> và <strong>cluster_rfm_stats.csv</strong>
                từ Google Drive sau khi chạy xong notebook Colab.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    seg_source   = f_seg
    stats_source = f_stats
    ksel_source  = f_ksel

# Load
with st.spinner("Đang tải dữ liệu..."):
    df_seg, df_stats, df_ksel = load_data(seg_source, stats_source, ksel_source)

# Ensure cluster column is int
df_seg["cluster"] = df_seg["cluster"].astype(int)
df_stats["cluster"] = df_stats["cluster"].astype(int)

# Gán CLUSTER_META theo chế độ dữ liệu
if data_mode == "⬆️ Upload file của bạn":
    CLUSTER_META = _build_cluster_meta(df_stats)
else:
    CLUSTER_META = CLUSTER_META_DEFAULT

# Compute scaler params & centroids from raw data
scaler_mean = df_seg[["recency","frequency","monetary"]].mean()
scaler_std  = df_seg[["recency","frequency","monetary"]].std()

centroids_scaled = (
    df_seg.groupby("cluster")[["r_scaled","f_scaled","m_scaled"]]
    .mean()
    .rename(columns={"r_scaled":"r","f_scaled":"f","m_scaled":"m"})
)

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Tổng quan",
    "🔵  Phân cụm 3D",
    "🔮  Dự đoán khách hàng",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – TỔNG QUAN
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-eyebrow">Tổng quan hệ thống</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Phân tích RFM – H&amp;M Customer Segments</div>', unsafe_allow_html=True)

    # KPI row
    n_customers  = len(df_seg)
    n_clusters   = df_seg["cluster"].nunique()
    total_revenue = df_stats["Total_Revenue"].sum()
    avg_frequency = df_seg["frequency"].mean()

    k1, k2, k3, k4 = st.columns(4)
    for col, label, val, sub in [
        (k1, "Tổng khách hàng",   f"{n_customers:,}",          "unique customer IDs"),
        (k2, "Số cụm (K)",        str(n_clusters),              "KMeans clusters"),
        (k3, "Tổng doanh thu",    f"${total_revenue:,.0f}",     "tổng monetary"),
        (k4, "Tần suất TB",       f"{avg_frequency:.1f}",       "lượt mua trung bình"),
    ]:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Cluster size bar chart ---
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="section-eyebrow">Phân bố cụm</div>', unsafe_allow_html=True)
        colors = [cluster_meta(c)["color"] for c in df_stats["cluster"]]
        labels = [cluster_meta(c)["label"]  for c in df_stats["cluster"]]
        fig_bar = go.Figure(go.Bar(
            x=[f"Cluster {c}<br><span style='font-size:11px'>{lbl}</span>"
               for c, lbl in zip(df_stats["cluster"], labels)],
            y=df_stats["Num_Customers"],
            marker_color=colors,
            text=df_stats["Pct_Customers"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Số KH: %{y:,}<extra></extra>",
        ))
        fig_bar.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="Số khách hàng", xaxis_title="",
            showlegend=False, height=320,
            margin=dict(t=20, b=10, l=10, r=10),
            font=dict(family="Inter", size=12),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-eyebrow">Doanh thu theo cụm</div>', unsafe_allow_html=True)
        fig_pie = px.pie(
            df_stats,
            values="Total_Revenue",
            names=[cluster_meta(c)["label"] for c in df_stats["cluster"]],
            color_discrete_sequence=[cluster_meta(c)["color"] for c in df_stats["cluster"]],
            hole=0.45,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            showlegend=False, height=320,
            margin=dict(t=20, b=10, l=10, r=10),
            font=dict(family="Inter", size=12),
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- RFM stats table ---
    st.markdown('<div class="section-eyebrow">Bảng thống kê RFM theo cụm</div>', unsafe_allow_html=True)
    display_stats = df_stats.copy()
    display_stats.insert(0, "Nhóm khách hàng",
                          display_stats["cluster"].apply(lambda c: cluster_meta(c)["label"]))
    display_stats = display_stats.rename(columns={
        "cluster": "Cluster",
        "Num_Customers": "Số KH",
        "Avg_Recency": "Recency TB (ngày)",
        "Avg_Frequency": "Frequency TB",
        "Total_Revenue": "Tổng doanh thu ($)",
        "Avg_Monetary": "Monetary TB ($)",
        "Pct_Customers": "% KH",
    })
    st.dataframe(
        display_stats,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Tổng doanh thu ($)": st.column_config.NumberColumn(format="$%.0f"),
            "Monetary TB ($)":    st.column_config.NumberColumn(format="$%.2f"),
            "% KH":               st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
        }
    )

    # --- K-selection chart (if available) ---
    if df_ksel is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-eyebrow">Elbow / Silhouette – Chọn K tối ưu</div>', unsafe_allow_html=True)
        fig_k = go.Figure()
        # Tìm cột K, WSSSE, Silhouette linh hoạt theo tên thực tế
        xcol  = next((c for c in df_ksel.columns if c.upper() == "K"), df_ksel.columns[0])
        wssse_col = next((c for c in df_ksel.columns if "WSSSE" in c.upper() or "INERTIA" in c.upper() or "PHÍ" in c or "CHI" in c.upper()), None)
        sil_col   = next((c for c in df_ksel.columns if "SILHOUETTE" in c.upper()), None)

        if wssse_col:
            fig_k.add_trace(go.Scatter(x=df_ksel[xcol], y=df_ksel[wssse_col],
                            mode="lines+markers", name="WSSSE",
                            line=dict(color="#1565c0", width=2),
                            marker=dict(size=7)))
        if sil_col:
            fig_k.add_trace(go.Scatter(x=df_ksel[xcol], y=df_ksel[sil_col],
                            mode="lines+markers", name="Silhouette",
                            yaxis="y2",
                            line=dict(color="#c06b72", width=2, dash="dot"),
                            marker=dict(size=7)))
            fig_k.update_layout(
                yaxis2=dict(title="Silhouette Score", overlaying="y", side="right", showgrid=False)
            )
        fig_k.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="K (số cụm)", yaxis_title="Inertia",
            height=300, margin=dict(t=20, b=10, l=10, r=10),
            font=dict(family="Inter", size=12),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_k, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – 3D CLUSTER VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-eyebrow">Không gian đặc trưng đã chuẩn hóa</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Phân cụm KMeans 3D – Recency / Frequency / Monetary</div>', unsafe_allow_html=True)

    # Sample for performance (max 5000 points)
    sample_size = min(5000, len(df_seg))
    df_plot = df_seg.sample(sample_size, random_state=42) if len(df_seg) > sample_size else df_seg.copy()

    df_plot["cluster_label"] = df_plot["cluster"].apply(lambda c: f"Cluster {c} – {cluster_meta(c)['label']}")
    color_map = {f"Cluster {c} – {cluster_meta(c)['label']}": cluster_meta(c)["color"]
                 for c in df_plot["cluster"].unique()}

    fig_3d = px.scatter_3d(
        df_plot,
        x="r_scaled", y="f_scaled", z="m_scaled",
        color="cluster_label",
        color_discrete_map=color_map,
        opacity=0.65,
        size_max=4,
        labels={"r_scaled": "Recency (scaled)",
                "f_scaled": "Frequency (scaled)",
                "m_scaled": "Monetary (scaled)",
                "cluster_label": "Cụm"},
        hover_data={"recency": True, "frequency": True, "monetary": ":.2f",
                    "r_scaled": False, "f_scaled": False, "m_scaled": False},
        title=f"Phân cụm 3D ({sample_size:,} điểm mẫu / {len(df_seg):,} khách hàng)",
    )

    # Add centroids
    for cid, row in centroids_scaled.iterrows():
        meta = cluster_meta(cid)
        fig_3d.add_trace(go.Scatter3d(
            x=[row["r"]], y=[row["f"]], z=[row["m"]],
            mode="markers+text",
            marker=dict(size=10, color=meta["color"], symbol="diamond",
                        line=dict(width=2, color="white")),
            text=[meta["label"]],
            textposition="top center",
            textfont=dict(size=10, color=meta["color"]),
            name=f"Centroid {cid}",
            showlegend=False,
            hovertemplate=f"<b>Centroid – {meta['label']}</b><extra></extra>",
        ))

    fig_3d.update_layout(
        height=600,
        margin=dict(t=50, b=10, l=10, r=10),
        font=dict(family="Inter", size=12),
        legend=dict(orientation="v", x=1.01, y=0.5,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e0e0e0", borderwidth=1),
        scene=dict(
            xaxis=dict(backgroundcolor="#f8f9fa", gridcolor="#dee2e6"),
            yaxis=dict(backgroundcolor="#f8f9fa", gridcolor="#dee2e6"),
            zaxis=dict(backgroundcolor="#f8f9fa", gridcolor="#dee2e6"),
        ),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # Cluster summary cards
    st.markdown('<div class="section-eyebrow">Đặc trưng từng cụm</div>', unsafe_allow_html=True)
    cols = st.columns(min(n_clusters, 5))
    for i, (_, srow) in enumerate(df_stats.sort_values("cluster").iterrows()):
        cid  = int(srow["cluster"])
        meta = cluster_meta(cid)
        col  = cols[i % len(cols)]
        col.markdown(f"""
        <div style="background:{meta['bg']}; border:1.5px solid {meta['color']}40;
                    border-radius:12px; padding:1rem; text-align:center;">
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                        color:{meta['color']}; font-size:0.95rem; margin-bottom:0.5rem;">
                Cluster {cid}<br>{meta['label']}
            </div>
            <div style="font-size:0.78rem; color:#444; line-height:1.7;">
                👥 <b>{int(srow['Num_Customers']):,}</b> KH ({srow['Pct_Customers']:.1f}%)<br>
                📅 Recency: <b>{srow['Avg_Recency']:.0f}</b> ngày<br>
                🔁 Frequency: <b>{srow['Avg_Frequency']:.1f}</b> lần<br>
                💰 Monetary: <b>${srow['Avg_Monetary']:,.0f}</b>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – DỰ ĐOÁN KHÁCH HÀNG MỚI
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-eyebrow">Phân loại khách hàng mới</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dự đoán Cluster từ chỉ số RFM</div>', unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1.2])

    with col_input:
        st.markdown("**Nhập chỉ số RFM của khách hàng:**")

        r_min, r_max = int(df_seg["recency"].min()), max(int(df_seg["recency"].max()), int(df_seg["recency"].min()) + 10)
        f_min, f_max = int(df_seg["frequency"].min()), max(int(df_seg["frequency"].max()), int(df_seg["frequency"].min()) + 10)
        m_min, m_max = float(df_seg["monetary"].min()), max(float(df_seg["monetary"].max()), float(df_seg["monetary"].min()) + 1.0)

        recency   = st.slider("📅 Recency (số ngày từ lần mua cuối)",
                              min_value=r_min, max_value=r_max, value=(r_min+r_max)//2,
                              help="Càng nhỏ = mua gần đây càng nhiều")
        frequency = st.slider("🔁 Frequency (số lần giao dịch)",
                              min_value=f_min, max_value=f_max, value=(f_min+f_max)//2)
        monetary  = st.slider("💰 Monetary (tổng chi tiêu $)",
                              min_value=float(m_min), max_value=float(m_max),
                              value=float((m_min+m_max)/2), format="$%.2f")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮  Dự đoán phân khúc", type="primary", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Các phân khúc khách hàng:**")
        for cid, meta in sorted(CLUSTER_META.items()):
            if cid >= n_clusters:
                continue
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.35rem;">
                <span class="badge" style="background:{meta['bg']}; color:{meta['color']};">
                    Cluster {cid}
                </span>
                <span style="font-weight:600; color:{meta['color']}; font-size:0.85rem;">{meta['label']}</span>
            </div>""", unsafe_allow_html=True)

    with col_result:
        if predict_btn:
            # Scale input (tránh chia cho 0 khi std = 0)
            r_s = (recency   - scaler_mean["recency"])   / (scaler_std["recency"]   if scaler_std["recency"]   != 0 else 1)
            f_s = (frequency - scaler_mean["frequency"]) / (scaler_std["frequency"] if scaler_std["frequency"] != 0 else 1)
            m_s = (monetary  - scaler_mean["monetary"])  / (scaler_std["monetary"]  if scaler_std["monetary"]  != 0 else 1)
            point = np.array([r_s, f_s, m_s])

            # Euclidean distance to each centroid
            dists = {}
            for cid, crow in centroids_scaled.iterrows():
                centroid = np.array([crow["r"], crow["f"], crow["m"]])
                dists[cid] = np.linalg.norm(point - centroid)

            best_cluster = min(dists, key=dists.get)
            meta = cluster_meta(best_cluster)

            st.markdown(f"""
            <div class="pred-box" style="background:{meta['bg']}; border:2px solid {meta['color']}60;">
                <div style="font-size:0.75rem; font-weight:700; letter-spacing:0.15em;
                            text-transform:uppercase; color:{meta['color']}; margin-bottom:0.3rem;">
                    Phân khúc dự đoán
                </div>
                <div class="pred-cluster" style="color:{meta['color']};">
                    Cluster {best_cluster}
                </div>
                <div class="pred-label" style="color:{meta['color']};">
                    {meta['label']}
                </div>
                <div style="margin-top:0.8rem; font-size:0.82rem; font-weight:500;
                            color:{meta['color']}; opacity:0.85;
                            border-top:1px solid {meta['color']}30; padding-top:0.6rem;">
                    💡 {meta.get('tip', '')}
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Khoảng cách tới tất cả centroid
            st.markdown("**Khoảng cách đến các centroid:**")
            dist_df = pd.DataFrame([
                {"Cluster": f"Cluster {c} – {cluster_meta(c)['label']}",
                 "Khoảng cách": round(d, 4)}
                for c, d in sorted(dists.items(), key=lambda x: x[1])
            ])
            st.dataframe(dist_df, hide_index=True, use_container_width=True)

            # Input recap
            st.markdown(f"""
            <div style="background:#f8f9fa; border-radius:10px; padding:1rem;
                        font-size:0.82rem; color:#555; margin-top:0.5rem;">
                <b>Input:</b> Recency = {recency} ngày &nbsp;|&nbsp;
                Frequency = {frequency} lần &nbsp;|&nbsp;
                Monetary = ${monetary:,.2f}
                <br>
                <b>Scaled:</b> R = {r_s:.3f} &nbsp;|&nbsp; F = {f_s:.3f} &nbsp;|&nbsp; M = {m_s:.3f}
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#f8f9fa; border-radius:14px; padding:3rem 2rem;
                        text-align:center; color:#aab5c2;">
                <div style="font-size:2.5rem; margin-bottom:0.8rem;">🔮</div>
                <div style="font-weight:600; font-size:1rem; color:#8a9bb0;">
                    Điền chỉ số RFM và nhấn<br><em>"Dự đoán phân khúc"</em>
                </div>
            </div>""", unsafe_allow_html=True)
