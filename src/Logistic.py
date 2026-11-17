from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)


BASE_DIR = Path(__file__).resolve().parent.parent

#24 đặc trưng
FEATURE_COLS = [
    'pts_per_game_zscore', 'ast_per_game_zscore', 'trb_per_game_zscore',
    'per_zscore', 'ws_zscore', 'vorp_zscore',
    'team_win_pct', 'prev_allstar_count',
    'mp_per_game', 'pts_per_game', 'ast_per_game', 'trb_per_game',
    'stl_per_game', 'blk_per_game', 'tov_per_game',
    'fg_percent', 'x3p_percent', 'ft_percent',
    'per', 'ts_percent', 'usg_percent', 'ws', 'bpm', 'vorp'
]



# nạp dữ liệu
def load_data():
    data_dir = BASE_DIR / "data" / "processed"
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, val_df, test_df

#huẩn luyện

def train_model(X_train, y_train):
    # 1. Baseline
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)

    # 2. Logistic Regression
    logreg = LogisticRegression(
        class_weight="balanced", # Cân bằng lớp thiểu số
        max_iter=1000,           # Vòng lặp tối đa
        solver="lbfgs",          # Thuật toán tối ưu
        random_state=42
    )
    logreg.fit(X_train, y_train)

    return dummy, logreg

#trích xuất trọng số
def explain_model(model, feature_names):
    df = pd.DataFrame({
        "Feature": feature_names,
        "Weight_Beta": model.coef_[0],
        "Odds_Ratio": np.exp(model.coef_[0])
    }).sort_values(by="Weight_Beta", ascending=False).reset_index(drop=True)
    return df


#đánh giá hiệu năng
def evaluate(model, X, y, split_name):
    y_prob = model.predict_proba(X)[:, 1] # Lấy cột xác suất nhãn 1
    y_pred = model.predict(X)

    print(f"\n--- ĐÁNH GIÁ TRÊN TẬP {split_name} ---")
    print(f"Precision : {precision_score(y, y_pred):.4f}")
    print(f"Recall    : {recall_score(y, y_pred):.4f} (Trúng {int(recall_score(y, y_pred)*y.sum())}/{y.sum()})")
    print(f"F1-Score  : {f1_score(y, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y, y_prob):.4f}")
    print(f"PR-AUC    : {average_precision_score(y, y_prob):.4f}")

#code chính
if __name__ == "__main__":
    print("Bắt đầu huấn luyện mô hình Logistic Regression...")
    
    # 1. Nạp dữ liệu
    train_df, val_df, test_df = load_data()
    target_col = "All_Star"
    
    # 2. Tách X và y
    X_train, y_train = train_df[FEATURE_COLS], train_df[target_col]
    X_val, y_val = val_df[FEATURE_COLS], val_df[target_col]
    X_test, y_test = test_df[FEATURE_COLS], test_df[target_col]
    
    # 3. Huấn luyện
    dummy, logreg = train_model(X_train, y_train)
    
    # 4. Xem các đặc trưng quan trọng nhất
    weights_df = explain_model(logreg, FEATURE_COLS)
    print("\nTop 5 đặc trưng thúc đẩy vào All-Star mạnh nhất:")
    print(weights_df.head(5))
    
    # 5. Đánh giá
    evaluate(logreg, X_val, y_val, "VALIDATION (2017-2019)")
    evaluate(logreg, X_test, y_test, "TEST (2020-2026)")
    
    # 6. Lưu mô hình
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(logreg, models_dir / "LogisticRegression.joblib")
    print("\nĐã lưu mô hình vào models/LogisticRegression.joblib thành công!")

    