# Bài kiểm tra trước và sau

Cùng một đề cho trước và sau khi học. 12 câu trắc nghiệm, mỗi câu 1 điểm, 10 phút, không tra cứu. Đáp án ở cuối file:
đừng đưa phần đó vào form.

## Nhiệm vụ học (25 phút, cả hai nhóm)

1. **Self-attention** (khoảng 9 phút): mỗi từ "nhìn" các từ khác ra sao, vì sao chia √d_k, mặt nạ nhân quả làm gì.
2. **CNN trên MNIST** (khoảng 8 phút): ảnh đi qua conv, ReLU và max-pool thế nào, kích thước mỗi lớp, tham số nằm ở đâu.
3. **Overfitting** (khoảng 8 phút): khi nào mô hình học thuộc, nhìn dấu hiệu ở đâu, weight decay giúp gì.

Nhóm A dùng trang (Transformer → Encoder → Self-Attention → Attention lõi; tab CNN; tab Huấn luyện với nhiễu nhãn
20%). Nhóm B dùng tài liệu tĩnh có cùng nội dung.

## Câu hỏi

**1.** Trong self-attention, output của một token là:
- a) token có điểm attention cao nhất
- b) tổng có trọng số của các vector value, trọng số là softmax của (query · key)
- c) trung bình cộng của mọi token
- d) vector key của chính nó

**2.** Vì sao chia điểm Q·Kᵀ cho √d_k?
- a) để tổng các trọng số bằng 1
- b) để điểm không lớn dần theo số chiều làm softmax bão hoà (gần one-hot, gradient rất nhỏ)
- c) để mô hình chạy nhanh hơn
- d) để bỏ các giá trị âm

**3.** Mặt nạ nhân quả (causal mask) trong decoder dùng để:
- a) bỏ các token đệm
- b) không cho mỗi vị trí nhìn các token phía sau nó
- c) giảm số tham số
- d) trộn thứ tự các token

**4.** Conv 3×3, không đệm, bước 1, trên ảnh 28×28 cho ra bản đồ cỡ:
- a) 28×28
- b) 27×27
- c) 26×26
- d) 25×25

**5.** Max-pool 2×2, bước 2 trên bản đồ 11×11 (như `nn.MaxPool2d`) cho ra:
- a) 6×6
- b) 5×5
- c) 5.5×5.5
- d) 11×11

**6.** Một lớp conv nhận 8 kênh, ra 16 kênh, kernel 3×3, có bias. Số tham số là:
- a) 16 × 9 = 144
- b) 8 × 16 = 128
- c) 16 × 8 × 9 + 16 = 1 168
- d) 16 × 8 × 9 = 1 152

**7.** Trong CNN nhỏ trên MNIST của trang (conv 8 kênh, conv 16 kênh, Linear 400 → 10), phần lớn tham số nằm ở:
- a) lớp conv đầu
- b) lớp conv thứ hai
- c) lớp Linear cuối
- d) các lớp max-pool

**8.** Vì sao cùng một bộ lọc conv trượt qua mọi vị trí lại hữu ích?
- a) để ảnh nhỏ lại
- b) một nét ở đâu cũng được nhận ra như nhau, và cần ít tham số hơn nhiều so với lớp kết nối đầy đủ
- c) để mạng không cần hàm kích hoạt
- d) để tránh overfitting hoàn toàn

**9.** Dấu hiệu rõ nhất của overfitting là:
- a) loss train cao
- b) loss train vẫn giảm trong khi loss test tăng lên, và loss test cao hơn hẳn loss train
- c) loss train dao động lên xuống
- d) accuracy trên tập train là 50%

**10.** Khi 20% nhãn của tập train bị gán sai và mạng rất lớn, train thật lâu sẽ:
- a) làm mạng học thuộc cả những nhãn sai, nên kết quả trên tập test tệ đi
- b) tự sửa các nhãn sai
- c) không thay đổi gì so với dữ liệu sạch
- d) làm loss train tăng lên

**11.** Weight decay giúp chống overfitting vì:
- a) nó tăng learning rate
- b) mỗi bước nó kéo trọng số về gần 0, buộc mạng chọn lời giải đơn giản hơn
- c) nó bỏ bớt dữ liệu
- d) nó tắt bớt neuron vĩnh viễn

**12.** Khi sinh văn bản, giảm nhiệt độ T xuống dưới 1 làm phân bố của token kế tiếp:
- a) phẳng hơn: các token hiếm dễ được chọn hơn
- b) nhọn hơn: gần với luôn chọn token có xác suất cao nhất
- c) không đổi
- d) bỏ hết các token có xác suất dưới 10%

## Mức tự tin (1–5, sau mỗi lần làm bài)

- Tôi giải thích được self-attention cho một người bạn.
- Tôi giải thích được ảnh đi qua một CNN ra sao.
- Tôi nhận ra được khi nào một mô hình overfit.

## Câu hỏi mở (chỉ sau khi học)

1. Điều gì giúp bạn hiểu nhất trong 25 phút vừa rồi?
2. Chỗ nào làm bạn rối hoặc hiểu sai?

---

## Đáp án (không đưa vào form)

| Câu | Đáp án | Ở đâu trong trang |
|---|---|---|
| 1 | b | Transformer → Encoder → Self-Attention → Attention lõi |
| 2 | b | khối "Chia √dₖ" bên trong Attention lõi |
| 3 | b | khối "Mặt nạ" của Decoder; bật/tắt mặt nạ trong phần số chạy thật |
| 4 | c | CNN (MNIST) → Conv 3×3, 8 kênh: ⌊(28 − 3)/1⌋ + 1 = 26 |
| 5 | b | CNN (MNIST) → MaxPool 2×2 (2): ⌊11/2⌋ = 5, hàng và cột thứ 11 bị bỏ |
| 6 | c | CNN (MNIST) → Conv 3×3, 16 kênh |
| 7 | c | CNN (MNIST), mục Tham số: 4 010 trên 5 258 |
| 8 | b | CNN (MNIST), mục "Vì sao như vậy" |
| 9 | b | Huấn luyện: nhiễu nhãn 20%, 16 neuron, xem đường loss train và test |
| 10 | a | Huấn luyện: lời khuyên "Overfitting" |
| 11 | b | Huấn luyện: thanh Weight decay λ, công thức trong panel |
| 12 | b | LLaMA 3 → Lấy mẫu token kế: kéo thanh nhiệt độ T |
