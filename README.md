[README.md](https://github.com/user-attachments/files/31690643/README.md)
# 🛍️ Phân Khúc Khách Hàng H&M – RFM Clustering với PySpark

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](https://btnhombigdata-nhom2.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/hanphn25611-rgb/big-data-baitapnhom-nhom2)

## Giới thiệu

Dự án xây dựng pipeline end-to-end phân khúc khách hàng quy mô lớn trên tập dữ liệu H&M Kaggle (~31.7 triệu giao dịch), sử dụng mô hình RFM kết hợp thuật toán KMeans của PySpark MLlib, và triển khai kết quả lên dashboard tương tác bằng Streamlit.

## Kiến trúc hệ thống

```
H&M Kaggle Dataset (31.7M giao dịch)
        ↓
Google Colab + PySpark  (Batch Processing – offline)
        ↓
3 file CSV Artifact
  ├── customer_segments.csv
  ├── cluster_rfm_stats.csv
  └── k_means_k_selection_summary.csv
        ↓
Streamlit App  (Serving Layer – online)
```

Lợi thế: App chỉ đọc CSV artifact — không phụ thuộc dữ liệu gốc. Khi có dữ liệu mới, chỉ cần chạy lại notebook → upload 3 CSV → dashboard tự cập nhật.

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Data Processing | PySpark / Spark SQL |
| Machine Learning | pyspark.ml – KMeans MLlib |
| Data Serving | CSV Artifact |
| Visualization | Streamlit + Plotly |
| Deployment | Streamlit Community Cloud |
| Version Control | GitHub |

## Cấu trúc thư mục

```
bt_nhom_bigdata/
├── data/
│   ├── customer_segments.csv          # Kết quả phân cụm từng khách hàng
│   ├── cluster_rfm_stats.csv          # Thống kê RFM theo từng cluster
│   └── k_means_k_selection_summary.csv # Dữ liệu Elbow/Silhouette chọn K
├── bt_nhom_bigdata_ver2.py            # Streamlit app
└── requirements.txt
```

## Tính năng của Dashboard

**Tab 1 – Tổng quan**
- KPI: tổng khách hàng, số cụm K, tổng doanh thu, tần suất trung bình
- Biểu đồ phân bố và doanh thu theo từng cluster
- Bảng thống kê RFM chi tiết
- Biểu đồ Elbow / Silhouette (khi có file `k_means_k_selection_summary.csv`)

**Tab 2 – Phân cụm 3D**
- Scatter plot 3D theo trục Recency / Frequency / Monetary (đã chuẩn hóa)
- Hiển thị centroid từng cluster
- Card đặc trưng từng phân khúc

**Tab 3 – Dự đoán khách hàng**
- Nhập R, F, M → dự đoán cluster tức thì bằng khoảng cách Euclidean đến centroid
- Gợi ý chiến lược CRM theo phân khúc

## Chế độ dữ liệu

| Chế độ | Mô tả |
|---|---|
| Dữ liệu mặc định | Dùng 3 file CSV có sẵn trong repo (K=5, ~3.234 KH H&M) |
| Upload file của bạn | Upload CSV từ pipeline riêng → app tự phát hiện K và gán nhãn cluster tự động |

## Hướng dẫn tái tạo pipeline

1. Mở notebook trên Google Colab https://colab.research.google.com/drive/1DZUNqdB_dQC4Yf8aiAHkFwfXY_mqVnBq?usp=sharing&authuser=1
2. Chạy các cell theo thứ tự để tính toán RFM, chọn K tối ưu (xem biểu đồ Elbow/Silhouette), huấn luyện KMeans
3. Tải xuống 3 file CSV artifact từ `/content/`
4. Upload lên Streamlit app qua sidebar → chọn **"Upload file của bạn"**

## Kết quả phân khúc (K=5, dữ liệu mặc định)

| Cluster | Nhóm khách hàng | Tỷ lệ | Đặc trưng |
|---|---|---|---|
| 0 | New / Inactive | ~63% | F thấp, R thấp |
| 1 | Loyal Customers | ~10% | F trung bình, M ổn định |
| 2 | Super VIP | ~0.1% | F=68 (outlier), M cao |
| 3 | Champions | ~2% | F và M cao |
| 4 | Potential Loyals | ~25% | Tiềm năng tăng trưởng |

## Demo

🔗 **Live App:** [https://btnhombigdata-nhom2.streamlit.app/](https://btnhombigdata-nhom2.streamlit.app/)
