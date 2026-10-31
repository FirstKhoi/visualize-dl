# visualize-dl: Giải phẫu mạng nơ-ron

Trang web 3D (three.js) để khám phá các mạng nơ-ron theo từng khối: Transformer, LLaMA 3 8B, ViT-B/16, Swin-T, ResNet-50, ConvNeXt-T, DenseNet-121, EfficientNet-B0, LSTM, U-Net, Stable Diffusion v1.5 và một lab train MLP ngay trong trang. Mỗi khối có shape, công thức, ví dụ số, số tham số, code PyTorch và paper gốc. Giao diện và nội dung bằng tiếng Việt; code, comment và commit bằng tiếng Anh.

Demo: https://firstkhoi.github.io/visualize-dl/ (GitHub Pages phục vụ `docs/` trên `main`). Chủ repo là sinh viên đang xây portfolio cho hồ sơ thạc sĩ/tiến sĩ MIT. Phần train mô hình bằng PyTorch nên để chủ repo tự viết; Claude lo phần tích hợp vào trang.

## Lệnh

```sh
npm test                                     # unit test (Node 22+, không cần cài gì), ~30 s
npm ci && npx playwright install chromium    # một lần, cho test trình duyệt
npm run test:e2e                             # test trong Chromium headless, chạy offline, ~90 s
./build-pages.sh                             # build lại docs/ — bắt buộc sau mọi thay đổi ở file của trang
node scripts/audit-layout.mjs --fonts        # mở mọi khối ở 5 cỡ màn hình, báo chữ bị che, panel cuộn ngang, chữ bị cắt; vài phút
node scripts/record-demo.mjs                 # quay lại media/demo.gif (cần ffmpeg)
node scripts/i18n.mjs                        # bản tiếng Anh đã phủ được bao nhiêu
```

Mở `index.html` trực tiếp bằng trình duyệt (`file://`) là chạy được: không có bước build. CI (`.github/workflows/ci.yml`) chạy cả hai bộ test, và fail nếu `docs/` khác với kết quả của `build-pages.sh`.

## Kiến trúc

| File | Vai trò |
|---|---|
| `index.html` | Toàn bộ giao diện: CSS, cảnh 3D, panel, lab, các demo. Một `<script type="module">` lớn |
| `blocks.js` | Khối dùng chung (attention, FFN, các lớp norm…), danh sách paper `PAPER` (dùng qua `ref(key, note)`), chỉ mục khái niệm `CONCEPTS`, `textId` (khoá của bản dịch), `fmt` (số trong `\( \)`) và `fmtText` (số trong văn xuôi) |
| `live.js` | Mọi thứ thật sự tính toán: `rng`/`gauss` (có seed), `TOY` (attention đồ chơi), `TOYCONV`, `TRAIN` (MLP train trong trang), `GRAD` (thí nghiệm gradient), `SAMPLE` (lấy mẫu từ kế tiếp), `SWIN` (cửa sổ dịch đồ chơi), `DIFF` (khuếch tán 2D: lịch nhiễu của SD, bộ khử nhiễu chính xác cho hỗn hợp Gauss, DDIM, guidance), `CNN` (forward của CNN MNIST và tiền xử lý hình vẽ kiểu MNIST) |
| `models/*.js` | Mỗi file một kiến trúc, dựng cây khối bằng `n(name, cat, shape, info, more)` |
| `models/weights/mnist-cnn.js` | Trọng số của CNN MNIST (`MNIST_CNN`), do `training/train_mnist_cnn.py` sinh ra. Không sửa tay; train lại thì chạy script rồi `npm test` |
| `training/` | Script train bằng NumPy (nguồn của trọng số trong repo), bản PyTorch tương đương cho Colab, `results.json`. Dữ liệu MNIST tải về `training/data/` (gitignore) |
| `i18n/en.js` | Bản tiếng Anh `EN_TEXT`: khoá là `textId(văn bản tiếng Việt)`. Chỉ tải khi `?lang=en` |
| `scripts/i18n.mjs` | Độ phủ bản dịch theo tab; `missing <tab>` in ra các mục còn thiếu, điền sẵn tiếng Việt để dịch tại chỗ |
| `scripts/audit-layout.mjs` | Kiểm tra bố cục panel trên mọi khối: `--size phone,phoneLand`, `--model swin,cnn`, `--lang en`, `--fonts` (font thật, cần mạng). Exit 1 nếu có lỗi. Chạy sau mọi thay đổi lớn về nội dung hoặc CSS của panel |
| `scripts/record-demo.mjs` | Quay `media/demo.gif`: Transformer → attention, vẽ chữ số cho CNN, train lab. Mặc định tiếng Anh (`--lang vi` để quay bản tiếng Việt) |
| `docs/` | Bản cho GitHub Pages do `build-pages.sh` sinh ra. Không sửa tay |
| `tests/*.test.mjs` | Unit test. `tests/load.mjs` chạy các `<script src>` của `index.html` trong Node, biến `console.assert` thành lỗi thật |
| `tests/e2e/` | Test bằng Playwright. `browser.mjs` lấy three.js từ `node_modules` (phiên bản phải khớp import map) và chặn mọi request ra mạng. MathJax mặc định được giả lập; test về bố cục mở trang với `app.open(hash, { mathjax: true })` để có công thức thật. Font được giả lập, trừ khi `{ fonts: true }` (chỉ các script trong `scripts/` dùng) |

Script là classic script, không dùng ES module, để trang vẫn mở được qua `file://`. Các `const` ở cấp cao nhất của `blocks.js`/`live.js` là biến toàn cục mà `index.html` đọc trực tiếp.

### Khối có số chạy thật: bảng `LIVE` trong `index.html`
Một khối gắn `live: '<key>'` trong `models/*.js`. `LIVE[key]` khai báo ba thứ:
- `matrix(memo)` hoặc `image(node)`: thứ được vẽ trên khối 3D (`image` nhận node đang vẽ, nên một key có thể vẽ khác nhau theo cấu hình của khối, ví dụ `node.swin`, `node.diff`);
- `section()`: phần giải thích trong panel;
- `demo`: tên demo sở hữu khối, để biết khi nào cần vẽ lại (dùng qua `showsDemo(level, demo)`);
- `mount()` (tuỳ chọn): chạy sau khi panel dựng xong và khi đổi theme, để vẽ các canvas trong panel.

Không có `matrix` lẫn `image` thì khối chỉ có phần trong panel, không có sprite (gốc của tab CNN, khối ảnh vào và hai khối pool của nó: sprite trên các khối mỏng đứng sát nhau sẽ che nhau).

Các key hiện có:
- attention: `qk`, `scale`, `mask`, `softmax`, `sdpa`, `av`;
- tích chập: `conv`;
- lab: `W1`, `W2`, `W3`, `boundary`;
- sampling: `sample`, `sampleProbs`, `sampleDraws`;
- Swin: `swin` (`node.swin` = `{ shifted }`, `{ rolled }` hoặc `{ mask }`);
- khuếch tán: `diff` (`node.diff.at` = `start`, `now` hoặc `end`);
- CNN thật trên MNIST: `cnn` (gốc), `cnnIn`, `cnnC1`, `cnnP1`, `cnnC2`, `cnnP2`, `cnnOut`.

Thêm demo mới thì thêm một mục vào bảng này. `tests/e2e/live.test.mjs` tự mở một khối của mỗi key và kiểm tra phần trong panel; nếu class của section khác `live` thì cập nhật map `SECTION` trong file test đó.

### Bản tiếng Anh
- `?lang=en|vi` chọn ngôn ngữ và được nhớ trong `localStorage`; nút EN/VI tải lại trang với `?lang`, giữ hash. Module script `await` tải `i18n/en.js` trước khi dựng gì, chỉ khi là tiếng Anh.
- Nội dung model (tên khối, shape, `info`, công thức, code, nhãn link, paper, khái niệm) đi qua `tr(s)`: bản dịch trong `EN_TEXT[textId(s)]`, không có thì giữ tiếng Việt và panel ghi "Not translated yet". Chuỗi chỉ được coi là tiếng Việt nếu có chữ có dấu.
- Chuỗi giao diện và demo viết cả hai thứ tiếng ngay trong `index.html` bằng `L(vi, en)`; từ đồ chơi của demo (câu attention, token lấy mẫu, tên bộ dữ liệu) qua `gloss(w)`. Hàm nào gọi `L()` thì không được đặt biến cục bộ tên `L` (các hàm về level như `paintLive(L)` có, nên đừng gọi `L()` trong đó).
- Đã dịch: giao diện, Transformer, LLaMA 3, Huấn luyện, CNN (`DONE` trong `tests/i18n.test.mjs`). Sửa một văn bản tiếng Việt của các tab này thì id đổi: test báo mục cũ là "stale" và văn bản mới là "missing"; dịch lại rồi xoá mục cũ. Dịch thêm tab: `node scripts/i18n.mjs missing <tab>`, dán vào `i18n/en.js`, dịch, thêm tab vào `DONE`. Giá trị là chuỗi JS: dấu `\` của TeX viết `\\`.
- `tests/e2e/lang.test.mjs` kiểm tra không còn chữ tiếng Việt trên các tab đã dịch (trừ link sang tab chưa dịch).

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
- **Số trong văn xuôi dùng `fmtText`**, không dùng `fmt`: MathJax chỉ xử lý trong `\( \)`, nên `\,` ngoài đó hiện nguyên văn (test trong `tests/page.test.mjs`).
- **Tránh trùng class CSS** với các demo khác (ví dụ `.tok`, `.live label` đã có sẵn).
- **Bố cục panel:**
  - Nút gắn với một mục (như "Sao chép") đặt trên hàng tiêu đề qua tham số `tool` của `sec(h, body, tool)`, không định vị tuyệt đối đè lên nội dung.
  - Công thức nằm trong câu không xuống dòng được, nên `.panel p` tự cuộn ngang. Thứ gì rộng (bảng, lưới) thì bọc trong một khung có `overflow-x: auto`.
  - Mục tiêu là panel không bao giờ cuộn ngang, kể cả trên điện thoại dọc (390px) và xoay ngang (844×390, panel chỉ khoảng 300px). Test: `tests/e2e/panel.test.mjs`.

## Git

- Branch làm việc: `claude/charming-brown-ogcaqu`. Chỉ mở PR khi chủ repo yêu cầu; chủ repo tự merge.
- Nếu PR của branch đã merge: chạy `git fetch origin main && git checkout -B claude/charming-brown-ogcaqu origin/main`, rồi `git push --force-with-lease`.
- Cuối commit message thêm dòng `Co-Authored-By` như các commit trước.

## Trạng thái (cập nhật lần cuối 2026-10-31; sửa mục này khi tiến độ thay đổi)

**Đã xong**
- PR #1: sửa 6 bug của lab; thêm test Node, CI, README song ngữ, LICENSE MIT.
- PR #2: test e2e; sửa các lỗi nhỏ (▶ sau khi giải xong, điểm cuối đồ thị, `aria-live`, tooltip, tốc độ).
- PR #3 (merge vào `main` ngày 2026-10-21): bảng `LIVE` và `rng` dùng chung; overfitting trong lab (tập test, nhiễu nhãn, weight decay, mini-batch); demo sampling của LLaMA (nhiệt độ T, top-p, bốc thăm); đã merge `main` (DenseNet-121, EfficientNet-B0); nút "Sao chép" không còn che code; panel không còn cuộn ngang trên điện thoại.
- Commit thẳng lên `main` (theo yêu cầu chủ repo, 2026-10-19 đến 2026-10-21): DenseNet-121, EfficientNet-B0, hệ số Dice cho U-Net (đối chiếu với bài review CNN ảnh y khoa của Mienye và cộng sự, 2025); Swin-T với demo cửa sổ dịch; Stable Diffusion v1.5 (CLIP, U-Net, guidance, DDIM, VAE decoder) với demo khuếch tán 2D; đối chiếu Swin với slide và paper, thêm cách tra bảng bias và khối nổi Swin V2; ConvNeXt-T (lộ trình Bảng 10 từ ResNet-50, khối nổi ConvNeXt V2: GRN, FCMAE). Khối nổi (`float: true`) đặt nhãn phía trên hộp. Số tham số của mọi model mới được assert so với torchvision/diffusers. Đường tắt residual vòng lên trên heatmap thay vì cắt ngang.
- Trên branch, chưa có PR (2026-10-31):
  - **CNN (MNIST)**: train bằng NumPy (`training/train_mnist_cnn.py`, 5 258 tham số, 98.23%), forward bằng JS khớp logit tới 1e-9, tab vẽ chữ số với feature map thật. Bản PyTorch (`training/mnist_cnn_pytorch.py`) chưa từng được chạy vì môi trường không có torch: lần đầu chạy trên Colab thì kiểm tra số.
  - **Cho thấy phần việc của chủ repo:** mục "How this was built" trong README, `training/README.md`, bản nháp bài viết `writing/lessons-draft.md` (chủ repo phải tự viết lại bằng lời của mình), form góp ý (`.github/ISSUE_TEMPLATE/feedback.yml`, link ở cuối mỗi khối), bộ nghiên cứu nhỏ `study/` (quy trình, bài kiểm tra 12 câu, bản nháp bài đăng).
  - **Bản tiếng Anh** cho giao diện và 4 tab (xem mục "Bản tiếng Anh").
  - Sửa số kiểu `128\,256` hiện nguyên văn trong văn xuôi (`fmtText`).
  - `scripts/audit-layout.mjs` (kiểm tra bố cục; lần chạy 2026-10-31 với font thật: không lỗi nào ở cả 5 cỡ màn hình, 1 017 panel khác nhau, tiếng Việt và 4 tab tiếng Anh) và `scripts/record-demo.mjs`; `media/demo.gif` quay lại bằng bản tiếng Anh (28 s, 3.8 MB).

**Bước tiếp theo, theo thứ tự**
1. **Việc của chủ repo (Claude không làm thay được):** viết lại bài viết bằng lời của mình; sửa mô tả "About" của repo trên GitHub (nhắc tới bản tiếng Anh và CNN thật); đăng bài giới thiệu (`study/announcement.md`); chạy nghiên cứu nhỏ (`study/README.md`), hỏi trường về hội đồng đạo đức nếu định công bố; chạy thử bản PyTorch trên Colab.
2. **GPT tí hon tiếng Việt** (train trên Truyện Kiều): attention và phân bố xác suất thật thay cho bản đồ chơi trong `TOY`/`SAMPLE`.
3. **Dịch 9 tab còn lại** (`node scripts/i18n.mjs` cho biết độ phủ), bắt đầu bằng ViT (ngắn nhất).
4. **Các demo còn lại của Giai đoạn 1:** trục chuẩn hoá (BN/LN/RMSNorm), chia ảnh thành patch cho ViT, receptive field cho ResNet, RoPE, BPE.
5. **MedMNIST + Grad-CAM** (CNN trên ảnh y khoa, nối với bài review của Mienye và cộng sự): chưa làm vì môi trường này không tải được dữ liệu từ Zenodo.
6. **MoE** (diffusion, CLIP, VAE đã có trong Stable Diffusion).

**Việc nhỏ còn tồn**
- **Unit test chậm (~30 s)** vì có hai lần train thật 2000 bước trong `train.test.mjs`.
- **Muốn bỏ `docs/` khỏi repo** thì phải đổi nguồn của Pages sang "GitHub Actions" trong cài đặt repo trước.
- **Bản trên claude.ai** (`ARTIFACT_URL`) cần đăng thêm `i18n/en.js` và `models/weights/mnist-cnn.js` khi đăng lại; thiếu `en.js` thì nút EN chỉ dịch giao diện.
