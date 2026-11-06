# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV và họ tên:** xem tên file báo cáo (theo mục 5 của `RULES.md`)
- **SEED:** 5317

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ (15/15 kiểm tra, đủ 10 dự đoán) |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

**Hạn chế:** LLM chạy bằng OpenAI (`gpt-6-luna`, `gpt-6.1-terra`) và chỉ đo trên SUB 40 tin, nên chênh lệch nhỏ có thể chỉ là nhiễu: giữa hai lần chạy cùng dữ liệu, F1 của 4a đổi từ 0.905 sang 0.950 dù `temperature=0`.

## 2. Kết quả chính

| Cách | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.865 | 0.919 | 0.049 | 0 |
| 2. BoW + MLP | 0.935 | 0.976 | 0.207 | 0 |
| 3a. Encoder + kNN | 0.896 | 1.000 | 33.462 | 0 |
| 3b. Encoder + LogReg | 0.952 | 1.000 | 13.420 | 0 |
| 3c. Encoder + MLP head | 0.937 | 0.947 | 12.104 | 0 |
| 4a. LLM nhỏ zero-shot | — | 0.950 | 1975.129 | 0.006 |
| 4b. LLM lớn zero-shot | — | 1.000 | 1962.031 | 0.060 |
| 4c. LLM nhỏ few-shot | — | 0.923 | 1795.818 | 0.019 |

Hàng LLM không có F1 test vì chỉ đo trên SUB. Hàng 4d (prompt phòng thủ) không có trong bảng vì chỉ chạy trên 50 case khó, xem mục 4.

## 3. Dự đoán và thực tế

1. **`ml_lr`, `lr=50`:** đoán loss dao động và cao đến cuối. Thực tế loss vọt lên khoảng 2.5 ở đầu rồi giảm còn 0.007. Lý do lệch: loss của logistic regression là hàm lồi, gradient bị chặn và sigmoid bão hoà làm bước thực tế nhỏ lại. Đây chỉ là loss train.
2. **`llm_models`, chi phí:** đoán model lớn đắt gấp khoảng 2.5 lần. Thực tế gấp 10 lần ($0.00241 so với $0.00024 cho 40 tin). Lý do lệch: dự đoán được viết theo giá Gemini (0.75 so với 0.30 USD/1M) khi notebook chưa chuyển sang OpenAI, và dự đoán này được giữ nguyên sau khi có kết quả. Phần dự đoán về F1 thì đúng: 0.950 so với 1.000, chênh 0.05 nên coi như hoà.
3. **`hard_eval`, LLM:** đoán accuracy khoảng 85% và khó nhất ở spam giọng tự nhiên. Thực tế 4a đạt 98%, 4d đạt 100%, và cả hai đúng 100% ở nhóm spam giọng tự nhiên. Lý do lệch: đánh giá thấp khả năng LLM đọc được ý định của cả câu. Bộ này chỉ có 50 case viết tay, nên kết quả không có nghĩa LLM không bao giờ sai: trên SUB, 4a vẫn sai 2/40 tin.

## 4. Phân tích 50 case khó

**BoW + LogReg tụt nhiều nhất:** F1 từ 0.919 xuống 0.286, accuracy 50%. Nhóm yếu nhất là spam giọng tự nhiên (0%) và thao túng (12%). Ngược lại, LLM không tụt: 4a đạt F1 0.983, accuracy 98%, chỉ sai 1 case ở nhóm "tưởng spam"; 4d đạt F1 1.000, đúng 50/50. Cả 4a và 4d đều đúng 100% ở nhóm prompt injection, nhưng 6 case là quá ít để kết luận model an toàn.

| Nhóm (% đoán đúng) | 1. BoW + LogReg | 2. BoW + MLP | 3a | 3b | 3c | 4a | 4d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tưởng spam nhưng không phải | 100 | 90 | 70 | 80 | 80 | 90 | 100 |
| Thao túng / lừa đảo | 12 | 38 | 88 | 38 | 25 | 100 | 100 |
| Spam giọng tự nhiên | 0 | 10 | 30 | 30 | 30 | 100 | 100 |
| Né từ khoá | 17 | 17 | 83 | 83 | 83 | 100 | 100 |
| Prompt injection | 67 | 83 | 100 | 100 | 100 | 100 | 100 |
| Nói về spam | 100 | 100 | 25 | 50 | 50 | 100 | 100 |
| Tin giao dịch hợp lệ | 83 | 100 | 83 | 83 | 83 | 100 | 100 |
| **Tổng (50 case)** | **50** | **58** | **68** | **64** | **62** | **98** | **100** |

Hai case dưới đây dùng danh sách lỗi của lần chạy trước, vì lần chạy cuối chỉ in lỗi của 4d. Hai lần chạy có F1 và accuracy giống hệt nhau ở 5 model không dùng LLM.

- **Case #14** (`Hi, I'm working from home and making about £300 a day...`): cả 5 model không dùng LLM đều sai, còn 4a và 4d đều đúng. Tin không có từ khoá spam quen thuộc nên BoW không đủ tín hiệu để vượt `b = −2.95`, còn nghĩa của câu lại gần một câu chuyện cá nhân nên embedding cũng xếp vào ham.
- **Case #50** (`Haha my brother changed his name in my phone to 'FREE PRIZE WINNER'...`): 3c đoán spam vì vector nghĩa bị kéo về vùng trúng thưởng, không phân biệt được việc *nói về* cụm từ này với việc mời nhận thưởng. BoW + LogReg đúng ở nhóm này, nhưng chủ yếu vì nó nghiêng về ham. 4a và 4d đều đúng.

## 5. Thử thách cá nhân

**Đề #2:** giảm vocab còn 100 từ và tăng lên 5000 từ, train lại và báo F1 test. **Cách làm:** chạy `build_vocab`, `vectorize`, `train_logreg` với từng kích thước, giữ `lr=0.5`, 300 epoch, dùng biến riêng để không ghi đè model chính.

| Vocab | F1 train | Recall test | F1 test | Train (s) |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.842 | 0.667 | 0.790 | 0.07 |
| 2.000 | 0.906 | 0.771 | 0.865 | 1.31 |
| 5.000 | 0.909 | 0.760 | 0.859 | 3.36 |

**Giải thích:** 100 từ bị underfit vì chỉ còn từ phổ biến, F1 train cũng thấp. Từ 2.000 lên 5.000 từ, F1 test gần như không đổi (giảm 0.006, tương ứng 1 tin spam) vì các từ thêm vào đều hiếm, trong khi thời gian train tăng khoảng 2,5 lần.

## 6. Phương án cho sản phẩm thật

**Chọn 3b (Encoder + LogReg)** cho yêu cầu p95 < 50 ms/tin, 10 triệu SMS/ngày: p95 13.42 ms, F1 test 0.952, cao nhất trong các cách đo trên toàn bộ tập test. Cả 5 cách không dùng LLM đều đạt SLA; các hàng LLM thì không (p95 1796–1975 ms, gấp 36–40 lần mức cho phép).

**Đánh đổi:** trên CPU này 3b chỉ đạt 158 tin/s so với mức trung bình 116 tin/s, nên cần nhiều máy hoặc GPU cho giờ cao điểm. Điểm yếu là các tin khó: 3b chỉ đúng 64% trên 50 case khó và 30% ở nhóm spam giọng tự nhiên.

**Chi phí nếu LLM xử lý mọi tin** (30 ngày, ước tính): `gpt-6-luna` khoảng $1.810/tháng theo số đo của notebook, hoặc khoảng $2.550/tháng nếu tính theo token API báo (khoảng 65 input + 4 output mỗi tin); `gpt-6.1-terra` khoảng $18.100/tháng. Gọi tuần tự chỉ đạt khoảng 0.65 tin/s, nên phải gọi song song rất nhiều.

**Cách dùng LLM hợp lý:** làm nhánh phụ cho tin có p của 3b trong khoảng [0.3, 0.7], dùng prompt phòng thủ. Nếu nhánh này chiếm dưới 5% lưu lượng thì chi phí khoảng $90–$130/tháng với `gpt-6-luna`. Tỉ lệ 5% chỉ là giả định, chưa được đo, và spam mà 3b tự tin xếp vào ham sẽ không bao giờ tới LLM.

## 7. AI log và điều đã học

| Phần | Hỏi gì | Nhận gì | Sửa hoặc kiểm chứng gì |
| --- | --- | --- | --- |
| API | Nhờ hướng dẫn cách lấy API key | Claude hướng dẫn cách lấy API của OpenAI | Làm theo hướng dẫn |
| Report | Nhờ sửa report theo kết quả mới trong notebook | Claude cập nhật số liệu và nội dung các mục 1–7 | Đối chiếu số liệu với output notebook và `lab1_report.json` |

**Điều học viên hiểu ra hôm nay:** Các quy tắc khi làm việc với API và cách gọi API.
