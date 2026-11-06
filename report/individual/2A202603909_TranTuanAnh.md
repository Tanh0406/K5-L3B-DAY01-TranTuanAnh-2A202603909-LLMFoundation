# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV và họ tên:** xem tên file báo cáo (theo mục 5 của `RULES.md`)
- **SEED:** 5317

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ (chưa có số liệu LLM) |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ (15/15 kiểm tra, chưa có kết quả LLM) |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ học viên tự viết tay |

**Hạn chế:** lần chạy Gemini bị dừng vì lỗi 429 và model lớn chưa gọi được, nên chưa có số liệu LLM thật. Các hàng 4a–4d không có trong bảng dưới.

## 2. Kết quả chính

| Cách | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.865 | 0.919 | 0.098 | 0 |
| 2. BoW + MLP | 0.935 | 0.976 | 0.227 | 0 |
| 3a. Encoder + kNN | 0.896 | 1.000 | 26.689 | 0 |
| 3b. Encoder + LogReg | 0.952 | 1.000 | 13.628 | 0 |
| 3c. Encoder + MLP head | 0.937 | 0.947 | 12.980 | 0 |
| 4a–4d. LLM (Gemini) | chưa đo | chưa đo | chưa đo | chưa đo |

## 3. Dự đoán và thực tế


1. **`ml_lr`, `lr=50`:** đoán loss dao động và cao đến cuối. Thực tế loss vọt lên khoảng 2.5 ở đầu rồi giảm còn 0.007. Lý do lệch: loss của logistic regression là hàm lồi, gradient bị chặn và sigmoid bão hoà làm bước thực tế nhỏ lại. Đây chỉ là loss train.
2. **`dl_curve`, MLP so với LogReg:** đoán F1 hơn 0.02–0.05. Thực tế hơn 0.070 (0.935 so với 0.865). Lý do lệch: LogReg chưa hội tụ hẳn sau 300 bước. Bản bỏ ReLU cũng đạt 0.935 nên phần lớn chênh lệch nhiều khả năng đến từ cách tối ưu.
3. **`tradeoff`:** không đoán BoW + MLP nằm trên đường Pareto. Thực tế nó nằm trên đường này (p95 0.227 ms, F1 SUB 0.976), và còn lấn át 3c. Lý do lệch: đánh giá thấp mức MLP vượt LogReg, trong khi embedding phải tốn thời gian encode mỗi tin.

## 4. Phân tích 50 case khó

**BoW + LogReg tụt nhiều nhất:** F1 từ 0.919 xuống 0.286, accuracy 50%. Nhóm yếu nhất là spam giọng tự nhiên (0%) và thao túng (12%). Chưa có số liệu LLM để so sánh.

- **Case #14** (`Hi, I'm working from home and making about £300 a day...`): cả 5 model đều sai. Tin không có từ khoá spam quen thuộc nên BoW không đủ tín hiệu để vượt `b = −2.95`, còn nghĩa của câu lại gần một câu chuyện cá nhân nên embedding cũng xếp vào ham.
- **Case #50** (`Haha my brother changed his name in my phone to 'FREE PRIZE WINNER'...`): 3c đoán spam vì vector nghĩa bị kéo về vùng trúng thưởng, không phân biệt được việc *nói về* cụm từ này với việc mời nhận thưởng. BoW + LogReg đúng ở nhóm này, nhưng chủ yếu vì nó nghiêng về ham.

## 5. Thử thách cá nhân

**Đề #2:** giảm vocab còn 100 từ và tăng lên 5000 từ, train lại và báo F1 test. **Cách làm:** chạy `build_vocab`, `vectorize`, `train_logreg` với từng kích thước, giữ `lr=0.5`, 300 epoch, dùng biến riêng để không ghi đè model chính.

| Vocab | F1 train | Recall test | F1 test | Train (s) |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.842 | 0.667 | 0.790 | 0.11 |
| 2.000 | 0.906 | 0.771 | 0.865 | 1.53 |
| 5.000 | 0.909 | 0.760 | 0.859 | 3.81 |

**Giải thích:** 100 từ bị underfit vì chỉ còn từ phổ biến, F1 train cũng thấp. Từ 2.000 lên 5.000 từ, F1 test gần như không đổi (giảm 0.006, tương ứng 1 tin spam) vì các từ thêm vào đều hiếm, trong khi thời gian train tăng khoảng 2,5 lần.

## 6. Phương án cho sản phẩm thật

**Chọn 3b (Encoder + LogReg)** cho yêu cầu p95 < 50 ms/tin, 10 triệu SMS/ngày: F1 test 0.952 cao nhất, p95 13.63 ms. **Đánh đổi:** trên CPU này chỉ đạt 147 tin/s so với mức trung bình 116 tin/s, nên cần nhiều máy hoặc GPU cho giờ cao điểm. Điểm yếu vẫn là spam giọng tự nhiên (30% ở bộ 50 case khó). LLM ước tính khoảng $9.750/tháng với `gemini-3.5-flash-lite` nếu xử lý mọi tin (100 token input, 1 token output, chưa tính suy luận) và một lần gọi thử mất 1.83 giây, nên chỉ hợp làm nhánh phụ cho tin lưỡng lự. Số liệu LLM ở đây là ước tính, chưa đo trên toàn bộ SUB.

## 7. AI log và điều đã học

| Phần | Hỏi gì | Nhận gì | Sửa hoặc kiểm chứng gì |
| --- | --- | --- | --- |
| API |Nhờ hướng dẫn cách lấy API key | Claude hướng dẫn cách lấy API của OpenAI | Làm theo hướng dẫn |

**Điều học viên hiểu ra hôm nay: Các quy tắc khi làm việc với API và cách gọi API