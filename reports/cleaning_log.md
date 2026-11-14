# NBA All-Star Dataset - Cleaning Log

## 1. Data source
Kaggle dataset: nba-aba-baa-stats.

## 2. Season filtering
- Giữ các mùa từ 1980 đến 2026.
- Loại bỏ mùa 1999.
- Chỉ giữ dữ liệu NBA.

## 3. Duplicate handling
- Khử trùng lặp theo player_id và season.
- Ưu tiên dòng tổng kết TOT đối với cầu thủ chuyển đội.
- Nếu không có dòng tổng kết, giữ dòng đội bóng duy nhất.

## 4. Team features
- Tính team_win_pct = W / (W + L).
- Xác định main_team theo số trận thi đấu nhiều nhất.
- Nếu hòa số trận, sử dụng quy tắc chọn theo thứ tự alphabet.

## 5. Target label
- All_Star = 1 nếu cầu thủ được chọn tham gia All-Star.
- Bao gồm cả cầu thủ thay thế (replaced=True).

## 6. Historical feature
- prev_allstar_count đếm số lần All-Star trước mùa hiện tại.
- Bao gồm lịch sử trước năm 1980.
- Không tính lần lựa chọn trong chính mùa đang xét.

## 7. Filtering
- Giữ cầu thủ có G >= 15.
- Giữ cầu thủ có MP/G >= 10.
- Loại 4 trường hợp All-Star không đạt ngưỡng lọc.

## 8. Missing values
- x3p_percent: điền median theo mùa.
- ft_percent: điền median theo mùa.
- gs: tạo gs_missing để đánh dấu dữ liệu thiếu, sau đó điền gs = 0; 403 giá trị thiếu tập trung ở mùa 1980 và 1981.
- Các đặc trưng được sử dụng trong 4 cấu hình feature hiện không còn giá trị thiếu.

## 9. Advanced statistics
- Ghép PER, TS%, USG%, BPM, VORP và WS.
- Ghép theo player_id và season.

## 10. Seasonal normalization
- Tạo 9 đặc trưng Z-score theo mùa:
  PTS, AST, TRB, STL, BLK, MP, PER, WS và VORP.
- Z-score được tính trên toàn bộ cầu thủ trong từng mùa.

## 11. Temporal split
- Train: season <= 2016.
- Validation: 2017-2019.
- Test: 2020-2026.

## 12. Data quality exceptions
- Magic Johnson mùa 1992 có trong danh sách All-Star nhưng không có bản ghi thống kê nguồn.
- Không tự tạo bản ghi thống kê cho trường hợp này.
- Vẫn giữ lịch sử lựa chọn của cầu thủ trong prev_allstar_count.
- 51 trường hợp hòa số trận khi xác định main_team được xử lý bằng thứ tự alphabet.

## 13. Methodological limitations
- Median imputation và seasonal Z-score hiện được tính từ toàn bộ dữ liệu từng mùa.
- Khi huấn luyện, cần xây dựng pipeline chống leakage phù hợp.
- Đây là bài toán hồi cứu sử dụng thống kê cả mùa, không phải dự đoán trước kỳ All-Star.