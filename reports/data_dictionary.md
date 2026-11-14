# NBA All-Star Dataset - Data Dictionary

- Rows: 15,542
- Columns: 53
- Primary key: player_id + season

| column              | data_type   | description                                                              |   missing_values |
|:--------------------|:------------|:-------------------------------------------------------------------------|-----------------:|
| season              | int64       | Mùa giải NBA.                                                            |                0 |
| lg                  | object      | League của cầu thủ.                                                      |                0 |
| player              | object      | Tên cầu thủ.                                                             |                0 |
| player_id           | object      | Mã định danh cầu thủ.                                                    |                0 |
| age                 | float64     | Tuổi cầu thủ trong mùa giải.                                             |                0 |
| team                | object      | Mã đội bóng trong dữ liệu gốc.                                           |                0 |
| pos                 | object      | Vị trí thi đấu.                                                          |                0 |
| g                   | int64       | Số trận đã thi đấu.                                                      |                0 |
| gs                  | float64     | Số trận đá chính.                                                        |                0 |
| mp_per_game         | float64     | Số phút thi đấu trung bình mỗi trận.                                     |                0 |
| fg_per_game         | float64     | Số lần ném rổ thành công trung bình mỗi trận.                            |                0 |
| fga_per_game        | float64     | Số lần ném rổ mỗi trận.                                                  |                0 |
| fg_percent          | float64     | Tỷ lệ ném rổ thành công.                                                 |                0 |
| x3p_per_game        | float64     | Số cú ném 3 điểm thành công mỗi trận.                                    |                0 |
| x3pa_per_game       | float64     | Số lần ném 3 điểm mỗi trận.                                              |                0 |
| x3p_percent         | float64     | Tỷ lệ ném 3 điểm thành công.                                             |                0 |
| x2p_per_game        | float64     | Số cú ném 2 điểm thành công mỗi trận.                                    |                0 |
| x2pa_per_game       | float64     | Số lần ném 2 điểm mỗi trận.                                              |                0 |
| x2p_percent         | float64     | Tỷ lệ ném 2 điểm thành công.                                             |                0 |
| e_fg_percent        | float64     | Tỷ lệ ném rổ hiệu quả (Effective FG%).                                   |                0 |
| ft_per_game         | float64     | Số lần ném phạt thành công mỗi trận.                                     |                0 |
| fta_per_game        | float64     | Số lần ném phạt mỗi trận.                                                |                0 |
| ft_percent          | float64     | Tỷ lệ ném phạt thành công.                                               |                0 |
| orb_per_game        | float64     | Số rebound tấn công mỗi trận.                                            |                0 |
| drb_per_game        | float64     | Số rebound phòng thủ mỗi trận.                                           |                0 |
| trb_per_game        | float64     | Tổng rebound trung bình mỗi trận.                                        |                0 |
| ast_per_game        | float64     | Số kiến tạo trung bình mỗi trận.                                         |                0 |
| stl_per_game        | float64     | Số lần cướp bóng trung bình mỗi trận.                                    |                0 |
| blk_per_game        | float64     | Số lần chắn bóng trung bình mỗi trận.                                    |                0 |
| tov_per_game        | float64     | Số lần mất bóng trung bình mỗi trận.                                     |                0 |
| pf_per_game         | float64     | Số lỗi cá nhân trung bình mỗi trận.                                      |                0 |
| pts_per_game        | float64     | Số điểm trung bình mỗi trận.                                             |                0 |
| main_team           | object      | Đội chính được xác định theo số trận thi đấu.                            |                0 |
| abbreviation        | object      | Mã viết tắt đội bóng dùng để ghép dữ liệu.                               |                0 |
| team_win_pct        | float64     | Tỷ lệ thắng của đội bóng trong mùa giải.                                 |                0 |
| All_Star            | int64       | Nhãn nhị phân: 1 nếu được chọn All-Star, 0 nếu không.                    |                0 |
| prev_allstar_count  | int64       | Số lần được chọn All-Star trước mùa giải hiện tại.                       |                0 |
| per                 | float64     | Player Efficiency Rating.                                                |                0 |
| ts_percent          | float64     | True Shooting Percentage.                                                |                0 |
| usg_percent         | float64     | Usage Percentage.                                                        |                0 |
| bpm                 | float64     | Box Plus/Minus.                                                          |                0 |
| vorp                | float64     | Value Over Replacement Player.                                           |                0 |
| ws                  | float64     | Win Shares.                                                              |                0 |
| pts_per_game_zscore | float64     | Z-score PTS so với các cầu thủ cùng mùa.                                 |                0 |
| ast_per_game_zscore | float64     | Z-score AST so với các cầu thủ cùng mùa.                                 |                0 |
| trb_per_game_zscore | float64     | Z-score TRB so với các cầu thủ cùng mùa.                                 |                0 |
| stl_per_game_zscore | float64     | Z-score STL so với các cầu thủ cùng mùa.                                 |                0 |
| blk_per_game_zscore | float64     | Z-score BLK so với các cầu thủ cùng mùa.                                 |                0 |
| mp_per_game_zscore  | float64     | Z-score MP so với các cầu thủ cùng mùa.                                  |                0 |
| gs_missing          | int8        | Cờ đánh dấu gs bị thiếu trong dữ liệu nguồn (1=có thiếu, 0=không thiếu). |                0 |
| per_zscore          | float64     | Z-score PER so với các cầu thủ cùng mùa.                                 |                0 |
| ws_zscore           | float64     | Z-score WS so với các cầu thủ cùng mùa.                                  |                0 |
| vorp_zscore         | float64     | Z-score VORP so với các cầu thủ cùng mùa.                                |                0 |