# Nghiên cứu nhỏ: trang này có giúp người học hiểu hơn không?

*A small study kit: protocol, pre/post quiz (`quiz.md`) and announcement drafts (`announcement.md`), in Vietnamese
because the participants will be Vietnamese students.*

Mục tiêu: có **số liệu** cho câu hỏi "khám phá mạng theo từng khối, có số chạy thật, có giúp hiểu nhanh và đúng hơn đọc
sơ đồ tĩnh không?". Có số liệu thì có thể viết một bài ngắn, ví dụ nộp workshop
[VISxAI](https://visxai.io/) ở IEEE VIS (xem hạn nộp năm đó), hoặc đưa vào bài viết và hồ sơ.

## Thiết kế

- **Người tham gia:** 16–20 sinh viên đã học hoặc đang học một môn ML cơ bản. Tuyển qua lớp, CLB, nhóm học ML.
- **Hai nhóm, chia ngẫu nhiên** (tung đồng xu hoặc xen kẽ theo thứ tự đăng ký):
  - **Nhóm A** dùng trang https://firstkhoi.github.io/visualize-dl/.
  - **Nhóm B** đọc cùng nội dung ở dạng tĩnh: sơ đồ trong paper gốc và phần giải thích của trang được chép ra một file PDF, không có 3D, không có số chạy.
- **Cùng thời gian, cùng nhiệm vụ** cho cả hai nhóm: 25 phút, ba chủ đề.
  1. Self-attention (Transformer → Encoder → Self-Attention → Attention lõi);
  2. CNN và MNIST (tab CNN);
  3. Overfitting (tab Huấn luyện).
- **Đo:**
  - bài kiểm tra trước và sau (`quiz.md`, 12 câu, cùng một đề);
  - mức tự tin (1–5) cho từng chủ đề;
  - hai câu hỏi mở.

  Có thể thêm thang SUS (System Usability Scale) cho nhóm A.

## Quy trình cho mỗi người (khoảng 50 phút)

1. Giới thiệu và đồng ý tham gia (2 phút, đọc phần "Đồng ý tham gia" bên dưới).
2. Làm bài trước (10 phút, không tra cứu).
3. Học 25 phút với tài liệu của nhóm mình, theo ba nhiệm vụ trong `quiz.md`.
4. Làm bài sau (10 phút, cùng đề), mức tự tin, câu hỏi mở.
5. Cảm ơn; với nhóm B, cho dùng thử trang sau khi xong.

Có thể chạy online qua Google Meet và Google Forms: chép câu hỏi trong `quiz.md` sang Google Forms, bật chế độ bài kiểm tra để chấm tự động.

## Phân tích

- **Chỉ số chính:** điểm tăng (sau − trước) của từng người, so giữa hai nhóm. Với 8–10 người mỗi nhóm, dùng kiểm định Mann–Whitney U, và báo cả trung vị, khoảng tin cậy và cỡ hiệu ứng, không chỉ giá trị p.
- **Trong nhóm A:** so điểm trước và sau (Wilcoxon cặp).
- **Câu hỏi mở:** gom thành vài chủ đề (điều gì giúp hiểu, chỗ nào gây rối). Đây thường là phần cho nhiều ý tưởng sửa trang nhất.
- **Viết rõ giới hạn:** mẫu nhỏ, người tham gia tự nguyện (có thể thích ML hơn trung bình), đo ngay sau khi học (chưa đo được nhớ lâu).

## Đạo đức và dữ liệu

- Tham gia tự nguyện, dừng lúc nào cũng được, không ảnh hưởng điểm số hay quan hệ gì.
- Không ghi tên vào dữ liệu: mỗi người một mã (A01, B01…). Chỉ lưu điểm và câu trả lời, không lưu email trong file kết quả.
- **Nếu làm trong trường đại học, hoặc định công bố kết quả:** hỏi giảng viên hoặc phòng quản lý khoa học xem có cần thông qua hội đồng đạo đức (IRB) không. Nhiều hội nghị yêu cầu điều này.
- Kết quả đã ẩn danh có thể để trong `study/results/`, kèm ghi chú cách thu thập.

### Đồng ý tham gia (đọc cho người tham gia)

> Bạn được mời tham gia một nghiên cứu nhỏ về cách học deep learning bằng hình ảnh, khoảng 50 phút. Bạn sẽ làm một
> bài kiểm tra ngắn, học 25 phút, rồi làm lại bài kiểm tra. Kết quả chỉ dùng để đánh giá và cải thiện một trang web
> học tập, và có thể được trình bày ở dạng tổng hợp, không có tên bạn. Bạn có thể dừng bất cứ lúc nào. Bạn đồng ý
> tham gia chứ?
