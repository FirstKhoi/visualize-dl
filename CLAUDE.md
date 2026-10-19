# visualize-dl: Giải phẫu mạng nơ-ron

Trang web 3D (three.js) để khám phá 7 mạng nơ-ron theo từng khối: Transformer, LLaMA 3 8B, ViT-B/16, ResNet-50, LSTM, U-Net và một lab train MLP ngay trong trang. Mỗi khối có shape, công thức, ví dụ số, số tham số, code PyTorch và paper gốc. Giao diện và nội dung bằng tiếng Việt; code, comment và commit bằng tiếng Anh.

Demo: https://firstkhoi.github.io/visualize-dl/ (GitHub Pages phục vụ `docs/` trên `main`). Chủ repo là sinh viên đang xây portfolio cho hồ sơ thạc sĩ/tiến sĩ MIT. Phần train mô hình bằng PyTorch nên để chủ repo tự viết; Claude lo phần tích hợp vào trang.

## Lệnh

```sh
npm test                                     # unit test (Node 22+, không cần cài gì), ~30 s
npm ci && npx playwright install chromium    # một lần, cho test trình duyệt
npm run test:e2e                             # test trong Chromium headless, chạy offline, ~90 s
./build-pages.sh                             # build lại docs/ — bắt buộc sau mọi thay đổi ở file của trang
```

Mở `index.html` trực tiếp bằng trình duyệt (`file://`) là chạy được: không có bước build. CI (`.github/workflows/ci.yml`) chạy cả hai bộ test, và fail nếu `docs/` khác với kết quả của `build-pages.sh`.

## Kiến trúc

| File | Vai trò |
|---|---|
| `index.html` | Toàn bộ giao diện: CSS, cảnh 3D, panel, lab, các demo. Một `<script type="module">` lớn |
| `blocks.js` | Khối dùng chung (attention, FFN, các lớp norm…), danh sách paper `PAPER` (dùng qua `ref(key, note)`), chỉ mục khái niệm `CONCEPTS` |
| `live.js` | Mọi thứ thật sự tính toán: `rng`/`gauss` (có seed), `TOY` (attention đồ chơi), `TOYCONV`, `TRAIN` (MLP train trong trang), `GRAD` (thí nghiệm gradient), `SAMPLE` (lấy mẫu từ kế tiếp) |
| `models/*.js` | Mỗi file một kiến trúc, dựng cây khối bằng `n(name, cat, shape, info, more)` |
| `docs/` | Bản cho GitHub Pages do `build-pages.sh` sinh ra. Không sửa tay |
| `tests/*.test.mjs` | Unit test. `tests/load.mjs` chạy các `<script src>` của `index.html` trong Node, biến `console.assert` thành lỗi thật |
| `tests/e2e/` | Test bằng Playwright. `browser.mjs` lấy three.js từ `node_modules` (phiên bản phải khớp import map), giả lập MathJax và font, và chặn mọi request ra mạng |

Script là classic script, không dùng ES module, để trang vẫn mở được qua `file://`. Các `const` ở cấp cao nhất của `blocks.js`/`live.js` là biến toàn cục mà `index.html` đọc trực tiếp.

### Khối có số chạy thật: bảng `LIVE` trong `index.html`
Một khối gắn `live: '<key>'` trong `models/*.js`. `LIVE[key]` khai báo ba thứ:
- `matrix(memo)` hoặc `image()`: thứ được vẽ trên khối 3D;
- `section()`: phần giải thích trong panel;
- `demo`: tên demo sở hữu khối, để biết khi nào cần vẽ lại (dùng qua `showsDemo(level, demo)`).

Các key hiện có:
- attention: `qk`, `scale`, `mask`, `softmax`, `sdpa`, `av`;
- tích chập: `conv`;
- lab: `W1`, `W2`, `W3`, `boundary`;
- sampling: `sample`, `sampleProbs`, `sampleDraws`.

Thêm demo mới thì thêm một mục vào bảng này. `tests/e2e/live.test.mjs` tự mở một khối của mỗi key và kiểm tra phần trong panel; nếu class của section khác `live` thì cập nhật map `SECTION` trong file test đó.

### API của `TRAIN` (live.js)
- `makeData(name, n = 200, { seed = 42, flip = 0 })`: `flip` là tỉ lệ nhãn bị lật (nhiễu nhãn); điểm không di chuyển. Tập test trong lab dùng `{ seed: 7 }`.
- `init(H, seed = 1)`, `resetOpt(net)`.
- `step(net, D, { act, opt, lr, decay = 0, batch = 0 })`: `batch = 0` là full batch. Weight decay tách rời kiểu AdamW, chỉ áp lên W, không áp lên bias. Trả `{ loss, acc }` của trọng số mới.
- `lossGrad` (backprop viết tay, có kiểm tra gradient bằng sai phân), `evaluate` (chỉ forward), `prober` (p(lớp 1) nhanh, dùng để vẽ ranh giới), `pacer(rate, most)`.
- `diagnose(h, acc, D, test?)` trả về một trong: `diverged | idle | dead | overfitting | solved | oscillating | slow | plateau | learning`. `index.html` (hàm `labAdvice`) đổi kết quả này thành lời khuyên.

## Quy ước

- **TDD:** viết test fail trước, nhìn nó fail đúng lý do, rồi mới sửa. Sau đó thử cố ý làm hỏng code để chắc test bắt được (mutation check). Kỳ vọng trong test lấy từ số tính tay, không tính bằng chính code đang test.
- **Con số viết trong bài phải đo thật**, và nếu được thì thêm `console.assert` trong script (test sẽ chạy nó). Ví dụ: số tham số của mọi model, ví dụ top-p trong `SAMPLE`, kiểm tra gradient của `TRAIN`.
- **Sửa `index.html`, `live.js`, `blocks.js`, `models/*`** thì chạy `./build-pages.sh` và commit cả `docs/`.
- **Phiên bản CDN được ghim:** `three@0.160.0` (devDependency phải trùng), `mathjax@3.2.2`, `es-module-shims@1.8.3`.
- **Node và Chromium lệch nhau ở chữ số cuối** sau khoảng 100 bước train, và Adam khuếch đại sai lệch đó. Không assert số chính xác của một lần chạy dài trong trình duyệt; ví dụ trong bài thì ghi theo khoảng.
- **Chromium headless trong môi trường cloud chỉ vẽ khoảng 5 khung hình/giây** (WebGL bằng phần mềm), nên lab chỉ chạy khoảng 25 bước/giây. Test e2e cần train lâu phải đặt timeout rộng. Trên máy thật, lab chạy 150–240 bước/giây.
- **Test e2e với phần tử bị vẽ lại liên tục:** Playwright không `hover` được vì phần tử không bao giờ "đứng yên". Hãy cuộn tới khi chưa chạy, rồi dùng `page.mouse.move` tới toạ độ cố định.
- **Tránh trùng class CSS** với các demo khác (ví dụ `.tok`, `.live label` đã có sẵn).

## Git

- Branch làm việc: `claude/charming-brown-ogcaqu`. Chỉ mở PR khi chủ repo yêu cầu; chủ repo tự merge.
- Nếu PR của branch đã merge: chạy `git fetch origin main && git checkout -B claude/charming-brown-ogcaqu origin/main`, rồi `git push --force-with-lease`.
- Cuối commit message thêm dòng `Co-Authored-By` như các commit trước.

## Trạng thái (cập nhật lần cuối 2026-10-19; sửa mục này khi tiến độ thay đổi)

**Đã xong**
- PR #1: sửa 6 bug của lab; thêm test Node, CI, README song ngữ, LICENSE MIT.
- PR #2: test e2e; sửa các lỗi nhỏ (▶ sau khi giải xong, điểm cuối đồ thị, `aria-live`, tooltip, tốc độ).
- PR #3 (đang mở lúc ghi): bảng `LIVE` và `rng` dùng chung; overfitting trong lab (tập test, nhiễu nhãn, weight decay, mini-batch); demo sampling của LLaMA (nhiệt độ T, top-p, bốc thăm).

**Bước tiếp theo, theo thứ tự**
1. **Giai đoạn 2: CNN thật trên MNIST.**
   - Chủ repo train trên Colab: conv 3×3 8 kênh → ReLU → maxpool → conv 3×3 16 kênh → ReLU → maxpool → FC 400→10. Khoảng 5.300 tham số, mục tiêu ≥ 98% trên tập test.
   - Export trọng số thành classic script, ví dụ `models/weights/mnist-cnn.js`, kèm 5 ảnh test và logit PyTorch tương ứng làm dữ liệu đối chiếu cho test.
   - Claude viết:
     - forward pass bằng JS, có test so với logit của PyTorch;
     - tab "CNN (MNIST)" với ô vẽ chữ số, thu về 28×28 và căn giữa theo trọng tâm như MNIST;
     - feature map thật trên khối 3D qua các key mới trong `LIVE`.
2. **GPT tí hon tiếng Việt** (train trên Truyện Kiều): attention và phân bố xác suất thật thay cho bản đồ chơi trong `TOY`/`SAMPLE`.
3. **Các demo còn lại của Giai đoạn 1:** trục chuẩn hoá (BN/LN/RMSNorm), chia ảnh thành patch cho ViT, receptive field cho ResNet, RoPE, BPE.
4. **Giai đoạn 3–4:** diffusion, CLIP, MoE, VAE; bản tiếng Anh (`info` → `{ vi, en }`); bài viết; nghiên cứu nhỏ với người dùng (workshop VISxAI).

**Việc nhỏ còn tồn**
- **`media/demo.gif` đã cũ.** Nên đưa script quay vào repo, ví dụ `scripts/record-demo.mjs`: Playwright `recordVideo`, rồi ffmpeg ra GIF 800px, 8 khung hình/giây, 64 màu, khoảng 3 MB.
- **Unit test chậm (~30 s)** vì có hai lần train thật 2000 bước trong `train.test.mjs`.
- **Muốn bỏ `docs/` khỏi repo** thì phải đổi nguồn của Pages sang "GitHub Actions" trong cài đặt repo trước.
