# Đề tài 1: Ước tính khối lượng hóa chất bị giấu (CBI) trong dữ liệu CDR 2024

**Tên dự kiến:** *Filling the confidentiality gap: machine-learning imputation of withheld chemical production volumes with aggregate-constrained reconciliation (US TSCA CDR 2024)*

## 1. Bài toán
- Trong 50.433 dòng (mỗi dòng là một cặp hóa chất × cơ sở), **21.624 dòng (43%) bị giấu khối lượng 2023** theo diện bí mật kinh doanh (CBI).
- Khối lượng toàn quốc của mỗi hóa chất vẫn được công bố dưới dạng khoảng giá trị.
- **Kiểm tra quan trọng:** với 1.387 hóa chất mà mọi cơ sở đều công khai, tổng khối lượng các cơ sở nằm trong khoảng toàn quốc ở **99,8%** trường hợp. Như vậy khoảng toàn quốc là một ràng buộc tổng hợp hợp lệ để hiệu chỉnh dự đoán.

## 2. Các phát hiện chính
1. **Dòng bị giấu khác hẳn dòng công khai.** Một bộ phân loại tách được hai loại dòng với AUC = 0,998 (chỉ số 0–1; 0,5 là đoán ngẫu nhiên, 1 là tách tuyệt đối). Sau khi đã giả lập kiểu giấu các trường khác trên dòng công khai, AUC vẫn là 0,994.
   - Dòng bị giấu khối lượng thường giấu luôn các trường khác: bang, ngành, công ty mẹ, số công nhân… với tỷ lệ 15–36%.
   - Yếu tố phân biệt mạnh nhất là **công ty**. 56% số dòng bị giấu (trong các dòng có tên công ty) đến từ các công ty giấu 100% số dòng của họ. Chỉ 10 tập đoàn đã chiếm 30% số dòng bị giấu.
2. **Chỉ dùng máy học thì bị sai lệch có hệ thống.** Khi kiểm định bằng cách để riêng từng công ty ra (leave-company-out), với nhóm 10% công ty lớn nhất mô hình **đoán thấp khoảng 4,5 lần** (sai lệch −0,65 log10).
3. **Hiệu chỉnh theo tổng toàn quốc sửa được sai lệch này.** Mô phỏng 5 lần, mỗi lần giả lập 20% số công ty chuyển sang giấu số liệu:

| Chỉ số (mức dòng) | Chỉ dùng ML | ML + hiệu chỉnh |
|---|---|---|
| Sai số tuyệt đối trung bình (log10) | 0,55 ± 0,03 | **0,32 ± 0,01** |
| Đúng hoặc lệch 1 nhóm | 86% | **96%** |
| Sai lệch trung bình (log10) | −0,24 | +0,12 |

4. **Kết quả kiểm định chéo** (nhóm chưa thấy tính theo hóa chất, dòng kiểm tra được giấu bớt trường theo kiểu CBI thật):

| Phương pháp | Sai số trung bình (log10) | Đúng nhóm | Đúng hoặc lệch 1 nhóm | Tương quan hạng (Spearman) |
|---|---|---|---|---|
| Chia đều khối lượng toàn quốc cho các cơ sở | 1,01 | 43% | 75% | 0,64 |
| LightGBM huấn luyện trên dữ liệu sạch | 0,57 | 53% | 92% | 0,82 |
| **LightGBM có tăng cường mặt nạ CBI** | **0,55** | **54%** | **93%** | **0,83** |

5. **Khoảng dự đoán 90% bằng CQR** (conformalized quantile regression): độ phủ thực tế 90,0%. Trên nhóm dòng công khai giống CBI nhất, độ phủ là 87,1%; với phiên bản có trọng số theo độ lệch phân phối, con số này tăng lên 88,9%.
6. **Áp dụng cho dòng bị giấu thật:**
   - 62% số dòng phải điều chỉnh **tăng** khi hiệu chỉnh theo tổng toàn quốc, với hệ số trung vị ×2,8. Điều này khớp với sai lệch đoán thấp ở phát hiện 2.
   - **Khoảng 41% tổng khối lượng hóa chất ở Mỹ nằm ở các dòng bị giấu.** Đây là cận dưới, vì khi tổng dự đoán thấp hơn khoảng toàn quốc, phép hiệu chỉnh chỉ kéo lên đến cận dưới của khoảng.
   - **PFAS:** 92% khối lượng nằm ở cơ sở bị giấu.
   - **20 hóa chất ưu tiên cao của TSCA:** formaldehyde 98%, 1,3-butadiene 81%, TBBPA 98%. Có 4 chất 100% bị giấu (xem Hình 2).

## 3. Đóng góp của bài
1. Lần đầu định lượng mức độ và cơ chế giấu số liệu trong CDR, cho thấy việc giấu chủ yếu là chính sách cấp công ty.
2. Đề xuất **tăng cường bằng mặt nạ CBI thực nghiệm**: lấy mẫu đúng tổ hợp các trường bị giấu từ dòng CBI thật để huấn luyện mô hình.
3. Đề xuất **hiệu chỉnh theo ràng buộc tổng hợp**, giúp giảm sai lệch do dữ liệu bị giấu không ngẫu nhiên.
4. Đưa ra khoảng dự đoán có bảo đảm thống kê (CQR và weighted CQR).
5. Công bố bộ dữ liệu khối lượng đã được bổ sung, phục vụ đánh giá phơi nhiễm và công bằng môi trường.

## 4. Dàn ý bài báo (nhắm ES&T, Environ. Modelling & Software hoặc Risk Analysis)
1. **Introduction:** dữ liệu bị giấu trong các hệ thống đăng ký hóa chất (TSCA, REACH); hệ quả cho đánh giá rủi ro; khoảng trống nghiên cứu.
2. **Data:** CDR 2024; kiểm tra tính khớp giữa tổng cơ sở và khoảng toàn quốc (99,8%).
3. **Methods:** chẩn đoán độ lệch phân phối (adversarial validation); LightGBM có tăng cường mặt nạ; CQR và weighted CQR; hiệu chỉnh theo ràng buộc tổng hợp; ba cách kiểm định (chưa thấy hóa chất, chưa thấy công ty, mô phỏng giấu theo công ty).
4. **Results:** cơ chế giấu số liệu; độ chính xác; độ phủ của khoảng dự đoán; ứng dụng cho PFAS và các chất ưu tiên TSCA.
5. **Discussion:** ý nghĩa chính sách (chỉnh sửa quy định CBI); khả năng áp dụng cho REACH và các kiểm kê khác; hạn chế.
6. **SI:** đặc trưng, siêu tham số, phân tích độ nhạy, toàn bộ bảng.

## 5. Hạn chế và việc cần làm tiếp
- [ ] **Mô phỏng hiện tại có thể lạc quan.** Nó chỉ dùng các hóa chất không có dòng CBI thật, và mỗi hóa chất thường chỉ có ít dòng bị giấu giả lập, nên phép hiệu chỉnh "biết" phần còn lại khá chính xác. Cần thêm kịch bản giấu nhiều cơ sở cho cùng một hóa chất, và kịch bản khoảng toàn quốc rộng.
- [ ] Với 26% dòng bị giấu cả danh tính cơ sở, ước tính chỉ dùng được ở cấp hóa chất, không định vị được trên bản đồ.
- [ ] Thêm đặc trưng cấu trúc phân tử (fingerprint, dùng OPSIN/RDKit đã cài) và giải thích mô hình bằng SHAP.
- [ ] Phép hiệu chỉnh đang co giãn tỷ lệ đều cho mọi dòng. Cần thử phân bổ có trọng số theo độ bất định, ví dụ phân bổ trong không gian log có trọng số bằng nghịch đảo độ rộng khoảng.
- [ ] Kiểm tra lại danh sách 20 chất ưu tiên cao của TSCA (tôi điền mã CAS từ trí nhớ; TCEP không xuất hiện trong dữ liệu).
- [ ] Tinh chỉnh siêu tham số và báo cáo độ biến thiên giữa các seed ngẫu nhiên.
- [ ] Nếu có thể, kiểm định độc lập bằng một nguồn khác, ví dụ đối chiếu với TRI ở cấp cơ sở (khối lượng phát thải tương quan với quy mô sản xuất).

## 6. File
| File | Nội dung |
|---|---|
| `01_build.py` | Dựng bảng dữ liệu cấp cơ sở và các kiểm tra ban đầu |
| `02_model.py` | Chẩn đoán độ lệch, mốc so sánh, LightGBM |
| `03_mask_aug.py` | Tăng cường mặt nạ, kiểm định khi giấu kiểu CBI, CQR |
| `03b_shift.py` | Đặc trưng gây lệch phân phối |
| `04_company_sim.py` | Kiểm định để riêng từng công ty và mô phỏng hiệu chỉnh |
| `05_apply.py` | Áp dụng cho dòng CBI thật và tổng hợp |
| `06_figs.py` | Vẽ hình |
| `results_*.json` | Toàn bộ số liệu kết quả |
| `imputed_cbi_rows.csv` | 21.624 dòng đã ước tính, kèm khoảng 90% |
| `chemical_level_summary.csv`, `tsca_high_priority_summary.csv` | Tổng hợp theo hóa chất |

Chạy theo thứ tự: `python 01_build.py <thư_mục_dữ_liệu> <thư_mục_đầu_ra>`, sau đó chạy các script 02–06 với tham số `<thư_mục_đầu_ra>`. Riêng `05_apply.py` cần thêm đường dẫn tới file phân loại PFAS.

---
## 7. Mô phỏng khó (`07_hard_sim.py`, `08_fig_hard.py`, Hình 3)
Chạy 3 kịch bản, mỗi kịch bản lặp 5 lần, so sánh 4 phương pháp:
- **A.** Giấu theo công ty.
- **B.** Tỷ lệ cơ sở bị giấu của mỗi hóa chất được lấy mẫu từ **phân bố thực tế** (kể cả trường hợp giấu 100%), ưu tiên giấu ở công ty lớn.
- **C.** Như B, nhưng khoảng toàn quốc bị làm thô thành khoảng rộng 10 lần.

Trong đó có một đối thủ mạnh không dùng máy học: **chia đều phần còn lại** (khoảng toàn quốc trừ tổng các dòng công khai, chia đều cho các cơ sở bị giấu).

Sai số MAE (log10) cho nhóm hóa chất **có một phần cơ sở công khai** (chiếm 74% dòng CBI thật):

| Kịch bản | Chia đều phần còn lại | Chỉ ML | **ML + hiệu chỉnh** |
|---|---|---|---|
| A | 0,40 | 0,48 | **0,38** |
| B | 0,39 | 0,48 | **0,36** |
| C (khoảng rộng) | 2,00 | 0,48 | **0,47** |

Với nhóm hóa chất **không có cơ sở công khai nào** (26%): B cho 0,26 / 0,34 / **0,22**; C cho 0,39 / 0,34 / **0,31**.

**Khoảng toàn quốc thực tế của các dòng CBI:** 75% có độ rộng ≤ 0,5 log10 (tức cận trên ≤ 3,2 lần cận dưới), 16% rộng 0,5–1, 9% là khoảng mở dạng "<X". Thực tế vì vậy gần với kịch bản B hơn, nhưng có khoảng 25% dòng thuộc kiểu C.

**Kết luận:** máy học và ràng buộc tổng toàn quốc **bổ trợ cho nhau**.
- Khi khoảng hẹp, ràng buộc tổng là yếu tố quyết định; máy học cải thiện thêm khoảng 8–15%.
- Khi khoảng rộng, phương pháp chỉ dùng ràng buộc sụp đổ, còn máy học vẫn đứng vững.
- Kết hợp cả hai **luôn tốt nhất trong mọi kịch bản**.

**Hạn chế còn lại:** khả năng xếp hạng đúng thứ tự các cơ sở bị giấu trong cùng một hóa chất còn yếu (Spearman 0,27–0,37). Sau hiệu chỉnh còn sai lệch dương khoảng +0,12–0,15 log10.
