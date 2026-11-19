# NBA All-Star Prediction

Dự án Học máy (Machine Learning) xây dựng quy trình phân loại và xếp hạng nhằm dự đoán danh sách cầu thủ được lựa chọn vào đội hình NBA All-Star hàng năm dựa trên các chỉ số thống kê phong độ cá nhân và hiệu năng thi đấu.

---

## 1. Yêu cầu môi trường và Phiên bản thư viện

Để đảm bảo tính tái lập 100% kết quả thực nghiệm (reproducibility) khớp chính xác với `reports/predictions.csv` và số liệu trong Báo cáo tổng thể, hệ thống yêu cầu môi trường thực thi chuẩn sau:

- Python: `3.10.x` (khuyến nghị `Python 3.10.11`)
- Thư viện scikit-learn: `scikit-learn == 1.6.1` (Bắt buộc phiên bản này để tái lập chính xác phân phối ngẫu nhiên của Random Forest tại seed 42 và cơ chế gán trọng số lớp `class_weight='balanced'`).

Cài đặt các gói phụ thuộc qua `pip`:

```bash
pip install -r requirements.txt
```

Chi tiết các thư viện chính trong `requirements.txt`:
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `scikit-learn == 1.6.1`
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`
- `jupyter >= 1.0.0`
- `joblib >= 1.3.0`

---

## 2. Cấu trúc thư mục dự án

```text
nba-allstar-prediction/
├── data/
│   ├── raw/                  # Dữ liệu nguồn thô từ Kaggle và Basketball-Reference
│   └── processed/            # Dữ liệu sau tiền xử lý (train.csv, val.csv, test.csv)
├── notebooks/
│   ├── 01_data_cleaning.ipynb        # Tiền xử lý, khử khác biệt thời kỳ và trích xuất đặc trưng
│   ├── 02_logistic_regression.ipynb  # Huấn luyện và giải thích Mô hình 1 (Logistic Regression)
│   ├── 03_random_forest.ipynb        # Huấn luyện, tối ưu ngưỡng và phân tích MDI/Permutation (Random Forest)
│   └── 04_model_evaluation.ipynb     # Đánh giá benchmark đối sánh hệ thống, Top-K và phân tích lỗi
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Module xử lý dữ liệu và pipeline scikit-learn
│   ├── train.py              # Module tích hợp huấn luyện và xuất dự đoán
│   ├── train_logistic.py     # Quy trình huấn luyện độc lập cho Mô hình 1
│   ├── train_random_forest.py# Quy trình huấn luyện độc lập cho Mô hình 2
│   └── evaluate.py           # Module tính toán metrics, Top-K per season và phân tích sai lệch
├── figures/
│   ├── eda/                  # Biểu đồ phân tích khám phá dữ liệu (EDA)
│   └── evaluation/           # Biểu đồ đánh giá đối sánh mô hình (PR-AUC, ROC-AUC, Top-K)
├── models/                   # Trọng số các mô hình đã huấn luyện (.joblib)
├── reports/                  # Tệp trao đổi kết quả dự đoán (predictions.csv)
├── requirements.txt          # Danh sách thư viện phụ thuộc ghim phiên bản
├── .gitignore                # Cấu hình loại trừ tệp nhị phân và dữ liệu trung gian
└── README.md                 # Tài liệu kỹ thuật của dự án
```

---

## 3. Không gian đặc trưng và Phân chia dữ liệu

### 3.1. Phân chia dữ liệu theo thời gian (Temporal Split)
Để hạn chế rò rỉ dữ liệu xuyên thời gian (Temporal Data Leakage) và mô phỏng chính xác bài toán kiểm tra năng lực tổng quát hóa sang tương lai:
- **Tập Huấn luyện (Train):** Các mùa giải từ 1980 đến 2016 (11.473 quan sát, 912 All-Star, tỷ lệ dương 7,95%).
- **Tập Thẩm định (Validation):** Các mùa giải từ 2017 đến 2019 (1.187 quan sát, 80 All-Star, tỷ lệ dương 6,74%) dùng để tinh chỉnh siêu tham số và quét ngưỡng cắt tối ưu $\tau^*$.
- **Tập Kiểm tra (Test):** Các mùa giải từ 2020 đến 2026 (2.882 quan sát, 186 All-Star, tỷ lệ dương 6,45%) dùng để đánh giá độc lập một lần duy nhất.

### 3.2. Không gian 24 đặc trưng thống nhất (`FEATURE_COLS`)
Toàn bộ các mô hình trong hệ thống sử dụng chung một không gian ma trận 24 chiều làm đầu vào:
1. **Chỉ số Z-score chuẩn hóa theo mùa (6 đặc trưng):** `pts_per_game_zscore`, `ast_per_game_zscore`, `trb_per_game_zscore`, `per_zscore`, `ws_zscore`, `vorp_zscore`. Đại diện cho 6 trụ cột phong độ tương đối so với mặt bằng cùng mùa giải.
2. **Thống kê mỗi trận (7 đặc trưng):** `mp_per_game`, `pts_per_game`, `ast_per_game`, `trb_per_game`, `stl_per_game`, `blk_per_game`, `tov_per_game`. Phản ánh sản lượng và đóng góp trực tiếp trong trận đấu.
3. **Hiệu quả dứt điểm (3 đặc trưng):** `fg_percent`, `x3p_percent`, `ft_percent`.
4. **Chỉ số nâng cao (6 đặc trưng):** `per`, `ts_percent`, `usg_percent`, `ws`, `bpm`, `vorp`. Đo lường hiệu quả toàn diện và giá trị đóng góp chiến thắng.
5. **Bối cảnh đội bóng và Lịch sử (2 đặc trưng):** `team_win_pct` (thành tích tập thể), `prev_allstar_count` (quán tính đẳng cấp và danh tiếng tích lũy từ quá khứ).

---

## 4. Kiến trúc mã nguồn (`src/`)

### 4.1. `src/preprocessing.py`
- `load_and_filter_seasons(df, min_season=1980)`: Loại bỏ các mùa giải trước 1980 (kỷ nguyên chưa có vạch 3 điểm) và loại bỏ mùa 1999 (lockout không tổ chức All-Star Game).
- `deduplicate_traded_players(df)`: Xử lý cầu thủ đổi đội giữa mùa giải; ưu tiên giữ dòng thống kê tổng hợp `TOT`, `2TM`, `3TM` để đảm bảo mỗi cặp `(player, season)` là duy nhất.
- `add_season_zscores(df, feature_cols)`: Chuẩn hóa Z-score theo từng mùa giải (`groupby('season')`) nhằm triệt tiêu sự khác biệt về nhịp độ thi đấu (pace) và mặt bằng điểm số giữa các thời kỳ.
- `split_by_season(df, train_end=2016, val_end=2019)`: Thực hiện phân chia thời gian chống rò rỉ dữ liệu.
- `build_preprocessing_pipeline(numeric_features)`: Cung cấp scikit-learn `Pipeline` kết hợp `SimpleImputer(strategy='median')` và `StandardScaler()`.

### 4.2. `src/train_logistic.py`
- Khởi tạo và huấn luyện `DummyClassifier(strategy='prior')` làm mốc đối chứng nền tảng.
- Khởi tạo và huấn luyện `LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)`:
  - Cân bằng trọng số lớp $w_1 \approx 6,29$ và $w_0 \approx 0,54$ (tỷ số phạt $\approx 11,6$ lần) nhằm giải quyết mất cân bằng lớp.
  - Tối ưu hóa tựa Newton (L-BFGS) kết hợp chuẩn hóa Ridge ($L_2$).
  - *Ghi chú kỹ thuật về hội tụ:* Mô hình huấn luyện trực tiếp trên 24 biến nhằm giữ nguyên thang đo tự nhiên phục vụ diễn giải trực quan hệ số $\beta$ và Odds Ratio. Do thang đo giữa biến gốc và Z-score có sự khác biệt, bộ giải tiệm cận hội tụ tại 1.000 vòng lặp. Thực nghiệm đối chứng với pipeline có `StandardScaler` cho thấy mô hình hội tụ hoàn toàn sau 84 vòng lặp và AP chỉ lệch khoảng 0,002 (val AP 0,8643, test AP 0,8704), không làm thay đổi bất kỳ kết luận so sánh nào trong đề tài.
- `update_predictions_file()`: Cơ chế Smart Upsert ghi đè đúng các dòng thuộc mô hình `LogisticRegression` và `Dummy`, bảo toàn tuyệt đối kết quả của `RandomForest` trong `reports/predictions.csv`.

### 4.3. `src/train_random_forest.py`
- Cấu hình chuẩn: `RandomForestClassifier(n_estimators=100, max_features='sqrt', bootstrap=True, class_weight='balanced', random_state=42, n_jobs=-1)`.
- Kết hợp Bagging và Feature Subspace ($m = \lfloor\sqrt{24}\rfloor = 4$ biến ngẫu nhiên tại mỗi nút).
- Phân tích độ quan trọng đặc trưng bằng MDI (Mean Decrease in Impurity) và Permutation Feature Importance trên tập Validation.

### 4.4. `src/evaluate.py`
- `load_predictions(csv_path)`: Đọc và kiểm tra tính toàn vẹn 6 cột chuẩn hóa của `reports/predictions.csv`:
  ```text
  ['player', 'season', 'split', 'model', 'y_true', 'y_proba']
  ```
- `compute_classification_metrics(y_true, y_proba, threshold=0.5)`: Tính toán bộ metric chuẩn gồm Precision, Recall, F1-Score, PR-AUC (`average_precision_score`), và ROC-AUC (`roc_auc_score`).
- `compute_topk_accuracy_per_season(df, k_per_season)`: Tính độ chính xác Top-K theo từng mùa giải ($K = \sum y_{\text{true}}$ All-Star thực tế mỗi mùa). Sắp xếp theo xác suất giảm dần; trường hợp hòa điểm xác suất được phá theo thứ tự alphabet tên cầu thủ (`player`).
- `get_error_cases(df, threshold)`: Trích xuất danh sách False Positives (FP - báo động giả) và False Negatives (FN - bỏ sót tài năng) phục vụ mổ xẻ định tính.

---

## 5. Phân công nhiệm vụ và Quy trình thực hiện Notebooks

| STT | Thành viên thực hiện | Phạm vi phụ trách trong Báo cáo | Notebook tương ứng | Nhiệm vụ chính trong dự án |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Lê Quý Dũng** | **Chương 1:** Dữ liệu và Phân tích khám phá dữ liệu (EDA) | `notebooks/01_data_cleaning.ipynb` | Thu thập dữ liệu từ 4 bảng nguồn NBA, xử lý cầu thủ chuyển đội giữa mùa, chuẩn hóa Z-score theo mùa, phân chia tập Train/Val/Test chống rò rỉ thời gian và hợp nhất không gian 24 đặc trưng (`FEATURE_COLS`). |
| **2** | **Trương Đỗ Gia Hoàng** | **Chương 2 & Chương 3:** Mô hình Dummy Baseline và Hồi quy Logistic | `notebooks/02_logistic_regression.ipynb` | Xây dựng lý thuyết và kiểm chuẩn Dummy Classifier; phát triển mô hình Hồi quy Logistic (L-BFGS, điều chuẩn $L_2$, trọng số `balanced`), khảo sát ngưỡng vận hành tối ưu $\tau^* = 0,65$, phân tích hệ số $\beta$ và Odds Ratio. |
| **3** | **Võ Thái Bảo Long** | **Chương 4 & Chương 5:** Mô hình Rừng ngẫu nhiên và Đánh giá đối sánh hệ thống | `notebooks/03_random_forest.ipynb`<br>`notebooks/04_model_evaluation.ipynb` | Phát triển mô hình RandomForestClassifier (100 cây, $m = 4$), khảo sát cơ chế `class_weight`, phân tích MDI và Permutation Importance; thiết kế khung đánh giá đối sánh hệ thống (PR-AUC, ROC-AUC, Top-K per season, Two-Stage Pipeline proposal). |

---

## 6. Hướng dẫn chạy và Tái lập kết quả

### Bước 1: Huấn luyện Hồi quy Logistic và Dummy Baseline
```bash
python src/train_logistic.py
```
Lệnh này sẽ huấn luyện mô hình, lưu vào `models/LogisticRegression.joblib` và cập nhật kết quả dự đoán của Logistic/Dummy vào `reports/predictions.csv`.

### Bước 2: Huấn luyện Rừng ngẫu nhiên
```bash
python src/train_random_forest.py
```
Lệnh này sẽ huấn luyện Random Forest tại seed 42, lưu vào `models/RandomForest.joblib` và cập nhật các dòng dự đoán của RF vào `reports/predictions.csv`.

### Bước 3: Đánh giá và Xuất biểu đồ đối sánh hệ thống
Mở và chạy toàn bộ tế bào trong notebook:
```bash
jupyter notebook notebooks/04_model_evaluation.ipynb
```
Hoặc thực thi script đánh giá:
```bash
python src/evaluate.py
```
Toàn bộ biểu đồ đường cong PR, đường cong ROC, biểu đồ Top-K per season và ma trận nhầm lẫn sẽ được xuất đồng bộ vào thư mục `figures/evaluation/`.
