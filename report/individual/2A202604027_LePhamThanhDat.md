# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202604027**
- **Họ và tên: Lê Phạm Thành Đạt**
- **SEED** (in ở ô đầu notebook):3043

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*

## 3. Dự đoán và thực tế

Dự đoán 1 - Embedding tốt hơn BoW: Em dự đoán Encoder + LogReg sẽ đúng 5–6/6 tin, nhưng thực tế chỉ đúng 4/6, còn BoW đúng 5/6. Vì Embedding không phải lúc nào cũng hiểu đúng ngữ cảnh.
Dự đoán 2 - Learning rate 50: Em dự đoán loss sẽ dao động mạnh hoặc không hội tụ, nhưng thực tế loss giảm xuống 0.006. Vì learning rate lớn vẫn có thể hội tụ tùy dữ liệu.
Dự đoán 3 - Model LLM lớn tốt hơn: Em dự đoán model lớn có F1 cao hơn model nhỏ khoảng 0.02–0.05, nhưng cả hai đều đạt 1.0. Vì tập SUB chỉ có 40 tin và model nhỏ đã phân loại đúng tất cả.

## 4. Phân tích 50 case khó

BoW + LogReg tụt F1 nhiều nhất, từ 0.857 xuống 0.378. Nhóm thao túng/lừa đảo đặc biệt khó vì mô hình chỉ dựa vào từ khóa.
Case 1: Tin giả làm người thân nhờ chuyển £200. BoW dễ bỏ sót vì câu viết giống tin nhắn gia đình thông thường.
Case 2: Tin nhắn viết "C0ngr4tulat10ns! Y0u h4ve w0n..." sử dụng số thay chữ để né từ khóa, khiến BoW khó nhận diện.
Hai case minh họa cơ chế gây khó cho BoW, không khẳng định cả hai đều bị dự đoán sai trong lần chạy.

## 5. Thử thách cá nhân

Đề #2: Thay đổi kích thước vocab thành 100 và 5000 từ.
Cách làm: Dùng build_vocab(max_size), vector hóa dữ liệu, train lại Logistic Regression và đánh giá F1.
Kết quả:
Vocab 100: F1 = 0.809, Accuracy = 95.4%.
Vocab 5000: F1 = 0.873, Accuracy = 96.8%.
Giải thích: Vocab lớn hơn giúp mô hình nhận diện nhiều từ quan trọng, cải thiện F1. Tuy nhiên, vocab quá lớn có thể tăng chi phí tính toán và nguy cơ overfitting.

## 6. Phương án cho sản phẩm thật

Với 10 triệu SMS/ngày và yêu cầu p95 dưới 50 ms, em chọn Encoder + MLP head vì có F1 SUB khoảng 0.974 và nằm trong nhóm đáp ứng SLA theo biểu đồ.
Em chấp nhận chi phí tính toán embedding và vẫn có nguy cơ phân loại sai. Có thể kết hợp LLM để kiểm tra các tin khó, nhưng cần xử lý bất đồng bộ vì LLM có độ trễ cao hơn 50 ms.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI hỗ trợ gì | Em sửa / kiểm chứng gì |
|---|---|---|---|
| Phần 1 – ML | Cách viết BoW, Sigmoid, Gradient Descent | Viết code và giải thích công thức | Chạy thử, kiểm tra F1 và thay đổi vocab |
| Phần 2 – DL | Cách xây dựng và huấn luyện MLP | Viết code PyTorch, giải thích Backpropagation | Kiểm tra loss, phát hiện overfitting |
| Phần 3 – Embedding | Cách tính Cosine Similarity và kNN | Viết code và giải thích thuật toán | So sánh BoW đúng 5/6 và Encoder + LogReg đúng 4/6 |
| Phần 4 – LLM | Cách gọi Gemini API, zero-shot, few-shot | Viết code API và tạo prompt | Sửa lỗi 429, 400, 503; kiểm tra F1 |
| Phần 5 – Đánh giá | Giải thích Pareto, SLA và stress test | Phân tích kết quả, đề xuất mô hình | Kiểm tra biểu đồ và 50 case khó |
| Tổng kết | Hỗ trợ tính tay và nhận xét kết quả | Cung cấp công thức và gợi ý trả lời | Đối chiếu kết quả thực tế |
### Điều em học được
Ban đầu em nghĩ mô hình càng phức tạp thì càng chính xác, nhưng thực tế không phải lúc nào cũng thế. Em hiểu rằng cần đánh giá mô hình bằng nhiều tiêu chí như F1, độ trễ, chi phí và khả năng xử lý tình huống khó. Em học được cách sử dụng AI để hỗ trợ lập trình nhưng vẫn phải tự chạy thử và kiểm chứng kết quả.
