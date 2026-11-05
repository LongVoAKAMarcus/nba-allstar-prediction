"""
Mô hình 1: Hồi quy Logistic (Logistic Regression) - NBA All-Star Prediction.
Thành viên phụ trách: Người 2 (Linear Classification & Mathematical Modeling).

Chức năng:
1. Nạp dữ liệu huấn luyện (train.csv), kiểm định (val.csv), và kiểm thử (test.csv).
2. Huấn luyện Mô hình 1 (LogisticRegression) cùng Mô hình cơ sở kiểm chuẩn (DummyClassifier).
3. Phân tích hệ số trọng số Beta (Feature Weights / Odds Ratio) để giải thích đóng góp của từng chỉ số.
4. Đánh giá hiệu năng chi tiết (Precision, Recall, F1, PR-AUC, ROC-AUC).
5. Lưu trữ trọng số mô hình vào 'models/LogisticRegression.joblib'.
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
from sklearn.linear_model import LogisticRegression
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


def train_logistic_pipeline(
    train_path: Optional[Union[str, Path]] = None,
    val_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    predictions_path: Optional[Union[str, Path]] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Huấn luyện và đánh giá Mô hình 1: Logistic Regression.
    """
    train_p = Path(train_path or BASE_DIR / "data" / "processed" / "train.csv")
    val_p = Path(val_path or BASE_DIR / "data" / "processed" / "val.csv")
    test_p = Path(test_path or BASE_DIR / "data" / "processed" / "test.csv")
    m_dir = Path(models_dir or BASE_DIR / "models")
    pred_p = Path(predictions_path or BASE_DIR / "reports" / "predictions.csv")

    print("=" * 75)
    print("NGƯỜI 2: HUẤN LUYỆN MÔ HÌNH 1 - HỒI QUY LOGISTIC (LOGISTIC REGRESSION)")
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
    logreg = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs"
    )

    # 2. Huấn luyện
    print("\n[BƯỚC 1] Đang huấn luyện LogisticRegression (có phạt lớp thiểu số class_weight='balanced')...")
    dummy.fit(X_train, y_train)
    logreg.fit(X_train, y_train)
    print("[HOÀN TẤT] Huấn luyện thành công!")

    # 3. Phân tích hệ số trọng số Beta (Feature Importance / Weights)
    coef_df = pd.DataFrame({
        "feature": features,
        "weight": logreg.coef_[0],
        "odds_ratio": np.exp(logreg.coef_[0])
    }).sort_values("weight", ascending=False).reset_index(drop=True)

    print("\n[BƯỚC 2] Phân tích Top 5 đặc trưng thúc đẩy khả năng được chọn All-Star mạnh nhất:")
    for idx, row in coef_df.head(5).iterrows():
        print(f"  {idx+1}. {row['feature']:<22} | Trọng số Beta: {row['weight']:+.4f} | Odds Ratio: {row['odds_ratio']:.2f}x")

    # 4. Dự đoán và đánh giá
    print("\n[BƯỚC 3] Đánh giá hiệu năng trên tập Kiểm định Validation (2017 - 2019):")
    y_val_proba = logreg.predict_proba(X_val)[:, 1]
    y_val_pred = logreg.predict(X_val)

    prec = precision_score(y_val, y_val_pred, zero_division=0)
    rec = recall_score(y_val, y_val_pred, zero_division=0)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    pr_auc = average_precision_score(y_val, y_val_proba)
    roc_auc = roc_auc_score(y_val, y_val_proba)

    print(f"  + Precision: {prec:.4f}")
    print(f"  + Recall:    {rec:.4f}  (Bắt trúng {int(rec*y_val.sum())}/{y_val.sum()} All-Stars)")
    print(f"  + F1-Score:  {f1:.4f}")
    print(f"  + PR-AUC:    {pr_auc:.4f}")
    print(f"  + ROC-AUC:   {roc_auc:.4f}")

    # 5. Lưu mô hình ra đĩa
    m_dir.mkdir(parents=True, exist_ok=True)
    save_path = m_dir / "LogisticRegression.joblib"
    dummy_path = m_dir / "Dummy.joblib"
    joblib.dump(logreg, save_path)
    joblib.dump(dummy, dummy_path)
    print(f"\n[BƯỚC 4] Đã lưu trọng số mô hình vào: {save_path.name} ({save_path.stat().st_size:,} bytes)")

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
        make_pred_df(val_df, X_val, y_val, "val", "LogisticRegression", logreg),
        make_pred_df(test_df, X_test, y_test, "test", "LogisticRegression", logreg),
        make_pred_df(val_df, X_val, y_val, "val", "Dummy", dummy),
        make_pred_df(test_df, X_test, y_test, "test", "Dummy", dummy),
    ]
    new_preds = pd.concat(preds_list, ignore_index=True)

    updated_df = update_predictions_file(new_preds, pred_p)
    print(f"[BƯỚC 5] Cập nhật thành công vào '{pred_p.name}'. Tổng bản ghi trong báo cáo: {len(updated_df):,}")

    return {
        "model": logreg,
        "dummy": dummy,
        "coefficients": coef_df,
        "metrics_val": {"precision": prec, "recall": rec, "f1": f1, "pr_auc": pr_auc, "roc_auc": roc_auc}
    }


if __name__ == "__main__":
    train_logistic_pipeline()
