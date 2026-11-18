"""
Mô hình 2: Rừng Ngẫu Nhiên (Random Forest Classifier) - NBA All-Star Prediction.
Người thực hiện: Võ Thái Bảo Long (Non-linear Tree Ensemble & Evaluation).

Chức năng:
1. Nạp dữ liệu huấn luyện (train.csv), kiểm định (val.csv), và kiểm thử (test.csv).
2. Huấn luyện Mô hình 2 (RandomForestClassifier) cùng Mô hình cơ sở kiểm chuẩn (DummyClassifier).
3. Phân tích Độ quan trọng đặc trưng Gini (Mean Decrease in Impurity - MDI) để xếp hạng đóng góp.
4. Đánh giá hiệu năng chi tiết (Precision, Recall, F1, PR-AUC, ROC-AUC).
5. Lưu trữ cấu trúc rừng cây vào 'models/RandomForest.joblib'.
6. Cập nhật kết quả dự đoán vào 'reports/predictions.csv'.
"""

import sys
from pathlib import Path

# Cấu hình encoding UTF-8 cho Windows terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# 24 đặc trưng tiêu chuẩn
FEATURE_COLS = [
    'pts_per_game_zscore', 'ast_per_game_zscore', 'trb_per_game_zscore',
    'per_zscore', 'ws_zscore', 'vorp_zscore',
    'team_win_pct', 'prev_allstar_count',
    'mp_per_game', 'pts_per_game', 'ast_per_game', 'trb_per_game',
    'stl_per_game', 'blk_per_game', 'tov_per_game',
    'fg_percent', 'x3p_percent', 'ft_percent',
    'per', 'ts_percent', 'usg_percent', 'ws', 'bpm', 'vorp'
]


def update_predictions_file(new_preds_df: pd.DataFrame, file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Cập nhật kết quả dự đoán của mô hình vào file 'reports/predictions.csv' mà không làm mất kết quả của các mô hình khác.
    """
    f_path = Path(file_path)
    f_path.parent.mkdir(parents=True, exist_ok=True)
    if f_path.exists():
        existing_df = pd.read_csv(f_path)
        new_models = set(new_preds_df["model"].unique())
        retained_df = existing_df[~existing_df["model"].isin(new_models)]
        combined_df = pd.concat([retained_df, new_preds_df], ignore_index=True)
        combined_df = combined_df.sort_values(["split", "season", "player", "model"]).reset_index(drop=True)
        combined_df.to_csv(f_path, index=False)
        return combined_df
    else:
        new_preds_df.to_csv(f_path, index=False)
        return new_preds_df


def train_random_forest_pipeline(
    train_path: Optional[Union[str, Path]] = None,
    val_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    predictions_path: Optional[Union[str, Path]] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Huấn luyện và đánh giá Mô hình 2: Random Forest Classifier.
    """
    train_p = Path(train_path or BASE_DIR / "data" / "processed" / "train.csv")
    val_p = Path(val_path or BASE_DIR / "data" / "processed" / "val.csv")
    test_p = Path(test_path or BASE_DIR / "data" / "processed" / "test.csv")
    m_dir = Path(models_dir or BASE_DIR / "models")
    pred_p = Path(predictions_path or BASE_DIR / "reports" / "predictions.csv")

    print("=" * 75)
    print("HUẤN LUYỆN MÔ HÌNH 2 - RỪNG NGẪU NHIÊN (RANDOM FOREST CLASSIFIER) - VÕ THÁI BẢO LONG")
    print("=" * 75)

    train_df = pd.read_csv(train_p)
    val_df = pd.read_csv(val_p)
    test_df = pd.read_csv(test_p)

    features = [c for c in FEATURE_COLS if c in train_df.columns]
    target_col = "All_Star" if "All_Star" in train_df.columns else "is_all_star"

    X_train, y_train = train_df[features], train_df[target_col]
    X_val, y_val = val_df[features], val_df[target_col]
    X_test, y_test = test_df[features], test_df[target_col]

    print(f"Dữ liệu huấn luyện: {len(X_train):,} dòng | Validation: {len(X_val):,} dòng | Test: {len(X_test):,} dòng")
    print(f"Số lượng đặc trưng đầu vào: {len(features)}")

    # 1. Khởi tạo mô hình
    dummy = DummyClassifier(strategy="prior")
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    # 2. Huấn luyện
    print("\n[BƯỚC 1] Đang huấn luyện RandomForestClassifier (Ensemble 100 cây quyết định, class_weight='balanced')...")
    dummy.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    print("[HOÀN TẤT] Huấn luyện thành công!")

    # 3. Phân tích độ quan trọng đặc trưng Gini (MDI Feature Importance)
    importances = rf.feature_importances_
    feat_df = pd.DataFrame({
        "feature": features,
        "importance": importances,
        "importance_pct": importances * 100
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n[BƯỚC 2] Phân tích Top 5 đặc trưng quan trọng nhất theo độ giảm tạp chất Gini:")
    for idx, row in feat_df.head(5).iterrows():
        print(f"  {idx+1}. {row['feature']:<22} | MDI Importance: {row['importance']:.4f} ({row['importance_pct']:.2f}%)")

    # 4. Dự đoán và đánh giá
    print("\n[BƯỚC 3] Đánh giá hiệu năng trên tập Kiểm định Validation (2017 - 2019):")
    y_val_proba = rf.predict_proba(X_val)[:, 1]
    y_val_pred = rf.predict(X_val)

    prec = precision_score(y_val, y_val_pred, zero_division=0)
    rec = recall_score(y_val, y_val_pred, zero_division=0)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    pr_auc = average_precision_score(y_val, y_val_proba)
    roc_auc = roc_auc_score(y_val, y_val_proba)

    print(f"  [Ngưỡng mặc định 0.50]")
    print(f"  + Precision: {prec:.4f}  (Độ chính xác rất cao, ít cảnh báo giả)")
    print(f"  + Recall:    {rec:.4f}  (Bắt trúng {int(rec*y_val.sum())}/{y_val.sum()} All-Stars)")
    print(f"  + F1-Score:  {f1:.4f}")
    print(f"  + PR-AUC:    {pr_auc:.4f}")
    print(f"  + ROC-AUC:   {roc_auc:.4f}")

    # Tối ưu hóa ngưỡng quyết định (Threshold Tuning) trên Validation
    best_thresh, best_f1 = 0.50, f1
    for th in np.arange(0.20, 0.75, 0.05):
        th_pred = (y_val_proba >= th).astype(int)
        th_f1 = f1_score(y_val, th_pred, zero_division=0)
        if th_f1 > best_f1:
            best_f1 = th_f1
            best_thresh = float(th)

    opt_val_pred = (y_val_proba >= best_thresh).astype(int)
    opt_prec = precision_score(y_val, opt_val_pred, zero_division=0)
    opt_rec = recall_score(y_val, opt_val_pred, zero_division=0)

    print(f"\n  [Tối ưu hóa ngưỡng: tau* = {best_thresh:.2f}]")
    print(f"  + Precision: {opt_prec:.4f}")
    print(f"  + Recall:    {opt_rec:.4f}  (Bắt trúng {int(opt_rec*y_val.sum())}/{y_val.sum()} All-Stars, +{int(opt_rec*y_val.sum()) - int(rec*y_val.sum())} cầu thủ)")
    print(f"  + F1-Score:  {best_f1:.4f}  (Cải thiện +{best_f1 - f1:.4f} so với ngưỡng 0.50)")

    # 5. Lưu mô hình ra đĩa
    m_dir.mkdir(parents=True, exist_ok=True)
    save_path = m_dir / "RandomForest.joblib"
    dummy_path = m_dir / "Dummy.joblib"
    joblib.dump(rf, save_path)
    joblib.dump(dummy, dummy_path)
    print(f"\n[BƯỚC 4] Đã lưu cấu trúc rừng cây vào: {save_path.name} ({save_path.stat().st_size / (1024*1024):.2f} MB)")

    # 6. Tạo dataframe dự đoán xuất chuẩn 6 cột
    def make_pred_df(df_split, X_split, y_split, split_name, model_name, model_obj):
        proba = model_obj.predict_proba(X_split)[:, 1] if hasattr(model_obj, "predict_proba") else np.zeros(len(df_split))
        return pd.DataFrame({
            "player": df_split["player"].values,
            "season": df_split["season"].values,
            "split": split_name,
            "model": model_name,
            "y_true": y_split.values,
            "y_proba": proba
        })

    preds_list = [
        make_pred_df(val_df, X_val, y_val, "val", "RandomForest", rf),
        make_pred_df(test_df, X_test, y_test, "test", "RandomForest", rf),
        make_pred_df(val_df, X_val, y_val, "val", "Dummy", dummy),
        make_pred_df(test_df, X_test, y_test, "test", "Dummy", dummy),
    ]
    new_preds = pd.concat(preds_list, ignore_index=True)

    updated_df = update_predictions_file(new_preds, pred_p)
    print(f"[BƯỚC 5] Cập nhật thành công vào '{pred_p.name}'. Tổng bản ghi trong báo cáo: {len(updated_df):,}")

    return {
        "model": rf,
        "dummy": dummy,
        "feature_importances": feat_df,
        "optimal_threshold": best_thresh,
        "metrics_val": {"precision": prec, "recall": rec, "f1": f1, "pr_auc": pr_auc, "roc_auc": roc_auc},
        "metrics_val_optimal": {"precision": opt_prec, "recall": opt_rec, "f1": best_f1, "threshold": best_thresh}
    }


if __name__ == "__main__":
    train_random_forest_pipeline()
