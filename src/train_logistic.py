# -*- coding: utf-8 -*-
"""
DỰ ÁN: NBA ALL-STAR PREDICTION
MÔ HÌNH: Hồi quy Logistic (Logistic Regression)
MỤC ĐÍCH: File huấn luyện mô hình chuẩn, chú thích ngắn gọn, đúng trọng tâm để trả lời vấn đáp.
"""

import sys
from pathlib import Path

# Cấu hình hiển thị tiếng Việt trên terminal
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from typing import Any, Dict, Optional, Union
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

# 24 đặc trưng đo lường phong độ cầu thủ
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
    Cập nhật kết quả dự đoán vào reports/predictions.csv.
    Giữ nguyên dự đoán của các mô hình khác (như Random Forest) và chỉ cập nhật mô hình hiện tại.
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
    Quy trình huấn luyện và đánh giá Hồi quy Logistic:
    1. Đọc dữ liệu Train (mùa <= 2016), Val (2017-2019), Test (2020-2026).
    2. Huấn luyện DummyClassifier (Baseline) và LogisticRegression (class_weight='balanced').
    3. Trích xuất hệ số Beta & Odds Ratio để giải thích đóng góp của từng chỉ số.
    4. Đánh giá chất lượng bằng Precision, Recall, F1, PR-AUC, ROC-AUC.
    5. Lưu mô hình vào models/ và xuất file reports/predictions.csv.
    """
    # 1. Khởi tạo đường dẫn
    train_p = Path(train_path or BASE_DIR / "data" / "processed" / "train.csv")
    val_p = Path(val_path or BASE_DIR / "data" / "processed" / "val.csv")
    test_p = Path(test_path or BASE_DIR / "data" / "processed" / "test.csv")
    m_dir = Path(models_dir or BASE_DIR / "models")
    pred_p = Path(predictions_path or BASE_DIR / "reports" / "predictions.csv")

    print("=" * 70)
    print("HUẤN LUYỆN MÔ HÌNH: HỒI QUY LOGISTIC (LOGISTIC REGRESSION)")
    print("=" * 70)

    # 2. Nạp dữ liệu
    # [VẤN ĐÁP]: Chia theo mùa giải (thời gian) để tránh rò rỉ dữ liệu (Data Leakage),
    # mô phỏng đúng thực tế: dùng dữ liệu quá khứ để dự đoán tương lai.
    train_df = pd.read_csv(train_p)
    val_df = pd.read_csv(val_p)
    test_df = pd.read_csv(test_p)

    features = [c for c in FEATURE_COLS if c in train_df.columns]
    target_col = "All_Star" if "All_Star" in train_df.columns else "is_all_star"

    X_train, y_train = train_df[features], train_df[target_col]
    X_val, y_val = val_df[features], val_df[target_col]
    X_test, y_test = test_df[features], test_df[target_col]

    print(f"[*] Dữ liệu Train: {len(X_train):,} dòng | Val: {len(X_val):,} dòng | Test: {len(X_test):,} dòng")
    print(f"[*] Tỷ lệ All-Star trên tập Train: {y_train.mean():.2%} (Mất cân bằng lớp nghiêm trọng)")

    # 3. Khởi tạo mô hình
    # [VẤN ĐÁP]: 
    # - DummyClassifier: Làm chuẩn đối chứng (Baseline). Luôn đoán nhãn đa số (0).
    # - class_weight='balanced': Tự động tính trọng số phạt w1 = N / (2 * N1) để phạt nặng lỗi bỏ sót All-Star.
    # - solver='lbfgs': Thuật toán tối ưu hóa tựa Newton giải bài toán lồi nhanh và chuẩn xác.
    # - max_iter=1000: Đảm bảo thuật toán đủ vòng lặp để hội tụ.
    dummy = DummyClassifier(strategy="prior")
    logreg = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs"
    )

    # 4. Huấn luyện (Fit)
    print("\n[*] Đang huấn luyện DummyClassifier và LogisticRegression...")
    dummy.fit(X_train, y_train)
    logreg.fit(X_train, y_train)
    print("[+] Huấn luyện hoàn tất!")

    # 5. Phân tích tính khả giải (Explainability): Hệ số Beta & Odds Ratio
    # [VẤN ĐÁP]:
    # - Hệ số Beta (w): w > 0 làm tăng xác suất All-Star, w < 0 làm giảm.
    # - Odds Ratio = exp(w): Khi chỉ số tăng 1 đơn vị, tỷ lệ cược (Odds) nhân thêm exp(w) lần.
    coef_df = pd.DataFrame({
        "feature": features,
        "weight": logreg.coef_[0],
        "odds_ratio": np.exp(logreg.coef_[0])
    }).sort_values("weight", ascending=False).reset_index(drop=True)

    print("\n" + "-" * 70)
    print("TOP 5 CHỈ SỐ QUYẾT ĐỊNH SUẤT ALL-STAR MẠNH NHẤT:")
    for idx, row in coef_df.head(5).iterrows():
        print(f"  {idx+1}. {row['feature']:<22} | Beta: {row['weight']:>+7.4f} | Odds Ratio: {row['odds_ratio']:>6.2f}x")
    print("-" * 70)

    # 6. Đánh giá mô hình trên tập Validation (2017 - 2019)
    # [VẤN ĐÁP]: Vì dữ liệu mất cân bằng, chỉ số quan trọng nhất là Recall (bắt trúng bao nhiêu All-Star)
    # và PR-AUC / F1-Score, không dựa vào Accuracy (đoán toàn 0 cũng đạt Accuracy 93% nhưng vô dụng).
    y_val_proba = logreg.predict_proba(X_val)[:, 1]
    y_val_pred = logreg.predict(X_val)

    prec = precision_score(y_val, y_val_pred, zero_division=0)
    rec = recall_score(y_val, y_val_pred, zero_division=0)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    pr_auc = average_precision_score(y_val, y_val_proba)
    roc_auc = roc_auc_score(y_val, y_val_proba)

    print("\nKẾT QUẢ ĐÁNH GIÁ TRÊN TẬP VALIDATION (2017 - 2019):")
    print(f"  - Precision: {prec:.4f} (Độ chuẩn xác khi dự đoán All-Star)")
    print(f"  - Recall:    {rec:.4f} (Bắt trúng {int((y_val_pred * y_val).sum())}/{int(y_val.sum())} All-Stars thực tế)")
    print(f"  - F1-Score:  {f1:.4f} (Trung hòa giữa Precision và Recall)")
    print(f"  - PR-AUC:    {pr_auc:.4f} (Độ đo chính xác nhất cho dữ liệu mất cân bằng)")
    print(f"  - ROC-AUC:   {roc_auc:.4f} (Khả năng phân biệt All-Star và cầu thủ thường)")

    # 7. Lưu trữ mô hình ra đĩa
    m_dir.mkdir(parents=True, exist_ok=True)
    save_path = m_dir / "LogisticRegression.joblib"
    dummy_path = m_dir / "Dummy.joblib"
    joblib.dump(logreg, save_path)
    joblib.dump(dummy, dummy_path)
    print(f"\n[+] Đã lưu mô hình vào: {save_path.name}")

    # 8. Xuất file kết quả dự đoán chuẩn 6 cột
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
    print(f"[+] Đã cập nhật kết quả dự đoán vào: {pred_p.name} ({len(updated_df):,} bản ghi)")
    print("=" * 70 + "\n")

    return {
        "model": logreg,
        "dummy": dummy,
        "coefficients": coef_df,
        "metrics_val": {"precision": prec, "recall": rec, "f1": f1, "pr_auc": pr_auc, "roc_auc": roc_auc}
    }


if __name__ == "__main__":
    train_logistic_pipeline()
