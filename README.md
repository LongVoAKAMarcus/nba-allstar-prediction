# NBA All-Star Prediction 🏀

Dự án Machine Learning dự đoán danh sách cầu thủ được lựa chọn vào đội hình **NBA All-Star** hàng năm dựa trên các chỉ số thống kê cá nhân và hiệu năng thi đấu.

---

## 1. Cấu trúc thư mục dự án

```text
nba-allstar-prediction/
├── data/
│   ├── raw/                  # Chứa dữ liệu thô (nba_player_stats.csv, ...)
│   └── processed/            # Dữ liệu đã làm sạch & phân chia (train.csv, val.csv, test.csv)
├── notebooks/
│   ├── 01_data_cleaning.ipynb    # Tiền xử lý, khử khác biệt thời kỳ & pipeline
│   ├── 02_model_training.ipynb   # Huấn luyện 4 mô hình & xuất predictions.csv
│   └── 03_evaluation.ipynb       # Đánh giá metrics, Top-K per season & Error analysis
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Module xử lý dữ liệu và pipeline scikit-learn
│   ├── train.py              # Module định nghĩa mô hình, huấn luyện và xuất dự đoán
│   └── evaluate.py           # Module tính toán metrics, Top-K và phân tích lỗi
├── figures/
│   ├── eda/                  # Biểu đồ phân tích khám phá dữ liệu (EDA)
│   └── evaluation/           # Biểu đồ đánh giá mô hình (PR-AUC, ROC-AUC, Top-K)
├── models/                   # Chứa trọng số các mô hình đã huấn luyện (.joblib)
├── reports/                  # Chứa predictions.csv và báo cáo phân tích
├── requirements.txt          # Danh sách thư viện phụ thuộc
├── .gitignore                # Cấu hình bỏ qua file nhị phân / dữ liệu nặng
└── README.md                 # Tài liệu hướng dẫn dự án
```

---

## 2. Cài đặt môi trường

Cài đặt các gói thư viện cần thiết bằng `pip`:

```bash
pip install -r requirements.txt
```

Các thư viện chính bao gồm:
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `scikit-learn >= 1.3.0`
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`
- `jupyter >= 1.0.0`
- `joblib >= 1.3.0`

---

## 3. Kiến trúc mã nguồn (`src/`)

### 3.1. `src/preprocessing.py`
- `load_and_filter_seasons(df, min_season=1980)`: Lọc bỏ các mùa giải trước 1980 (kỷ nguyên trước vạch 3 điểm) nhằm đồng nhất đặc trưng chiến thuật.
- `deduplicate_traded_players(df)`: Xử lý cầu thủ đổi đội giữa mùa giải; ưu tiên giữ dòng thống kê tổng hợp `TOT`, `2TM`, `3TM` để đảm bảo mỗi cặp `(player, season)` là duy nhất.
- `add_season_zscores(df, feature_cols)`: Chuẩn hóa Z-score theo từng mùa giải (`groupby('season')`) nhằm triệt tiêu sự khác biệt về nhịp độ thi đấu (pace) và mặt bằng điểm số giữa các thời kỳ.
- `split_by_season(df, train_end=2016, val_end=2019)`: Chia dữ liệu theo mốc thời gian để chống rò rỉ dữ liệu (Temporal Data Leakage):
  - **Train:** Mùa giải $\le 2016$
  - **Validation:** $2017 \le \text{mùa giải} \le 2019$
  - **Test:** Mùa giải $\ge 2020$
- `build_preprocessing_pipeline(numeric_features)`: Xây dựng scikit-learn `Pipeline` / `ColumnTransformer` kết hợp `SimpleImputer(strategy='median')` và `StandardScaler()` (chỉ `fit` trên tập Train).

### 3.2. `src/train.py`
- Khởi tạo 4 mô hình phân loại:
  1. `DummyClassifier(strategy='prior')`: Mô hình baseline cơ sở.
  2. `LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)`: Mô hình tuyến tính dễ diễn giải, cân bằng nhãn thiểu số.
  3. `RandomForestClassifier(class_weight='balanced', random_state=42)`: Mô hình cây quyết định kết hợp ensemble bagging.
  4. `HistGradientBoostingClassifier(random_state=42)`: Mô hình gradient boosting hiệu năng cao.
- `train_models(models_dict, X_train, y_train)`: Huấn luyện đồng thời các mô hình trên tập Train.
- `export_predictions(models_dict, X_df, y_true, split_name, output_path)`: Xuất file dự đoán chuẩn hóa gồm đúng 6 cột:
  ```text
  ['player', 'season', 'split', 'model', 'y_true', 'y_proba']
  ```
- `save_models` / `load_models`: Quản lý lưu trữ và nạp các mô hình dưới dạng `.joblib`.

### 3.3. `src/evaluate.py`
- `compute_classification_metrics(y_true, y_proba, threshold=0.5)`: Tính Precision, Recall, F1-score, PR-AUC (`average_precision_score`), và ROC-AUC (`roc_auc_score`).
- `compute_topk_accuracy_per_season(pred_df)`: Với mỗi mùa giải, lấy $k = \sum y_{true}$ (tổng số All-Star thực tế của mùa đó, thông thường 24-28 cầu thủ), chọn ra $k$ cầu thủ có xác suất `y_proba` cao nhất và tính tỷ lệ trùng khớp (Top-K Hit Rate).
- `get_error_cases(pred_df, threshold=0.5)`:
  - **False Positives (FP):** Cầu thủ có xác suất cao nhưng không được chọn (cầu thủ overhyped hoặc bỏ phiếu không bầu).
  - **False Negatives (FN):** Cầu thủ All-Star thực tế nhưng mô hình dự đoán xác suất thấp (những trường hợp bị bỏ sót / All-Star snubs).
- `evaluate_all_models(pred_df, threshold=0.5, split)`: Tạo bảng xếp hạng so sánh hiệu năng tổng hợp giữa tất cả mô hình.

---

## 4. Phân công công việc & Quy trình Notebooks

| Notebook | Thành viên phụ trách | Nhiệm vụ chính |
| :--- | :--- | :--- |
| **`01_data_cleaning.ipynb`** | **Thành viên 1** (Data Engineering) | Tải dữ liệu, lọc mùa giải hiện đại, khử trùng lặp cầu thủ chuyển nhượng, chuẩn hóa Z-score theo mùa, chia tập dữ liệu và xây dựng tiền xử lý pipeline. |
| **`02_model_training.ipynb`** | **Thành viên 2** (Model Development) | Khởi tạo 4 mô hình phân loại, huấn luyện trên tập Train, dự đoán xác suất và xuất `reports/predictions.csv`, lưu trữ artifacts vào `models/`. |
| **`03_evaluation.ipynb`** | **Thành viên 3** (Performance Analysis) | Nạp `predictions.csv`, tính metrics phân loại, đánh giá độ chính xác Top-K theo từng mùa giải, phân tích trường hợp lỗi (FP/FN), và trực quan hóa biểu đồ. |
