// Shared building blocks for every model file. Classic script (not a module) so index.html also opens from file://.
// Each node: name, cat, shape, info {what, formula[], why, example, params, code}, optional children.
// Shapes drop the batch dimension B. Box size = output tensor: x = channels/features, y·z = spatial size.
// Transformer-family blocks take a config c = { d, h, dk, dff, act, drop, noMask }.
const MODELS = {};
const R = String.raw;
const n = (name, cat, shape, info, more = {}) => ({ name, cat, shape, info, ...more });
const fmt = v => String(v).replace(/\B(?=(\d{3})+(?!\d))/g, '\\,');
const sqrtTeX = v => Number.isInteger(Math.sqrt(v)) ? R`\sqrt{${v}} = ${Math.sqrt(v)}` : R`\sqrt{${v}} \approx ${Math.sqrt(v).toFixed(2)}`;
const sp = H => Math.min(2.8, Math.max(0.22, 0.55 + 0.42 * Math.log2(H / 7)));
const ch = C => Math.max(0.06, 0.1 + 0.2 * Math.log2(C / 16));
const vol = (C, H) => [ch(C), sp(H), sp(H)];
const SLAB = [0.3, 1.5, 1.3];
const BALL = [0.62, 0.62, 0.62]; // rendered as a sphere: element-wise add / multiply

// Papers cited by blocks (ref field). ref(key, where) → { t, u, note }.
const PAPER = {
  vaswani: ['Vaswani và cộng sự (2017), Attention Is All You Need', 'https://arxiv.org/abs/1706.03762'],
  flash: ['Dao và cộng sự (2022), FlashAttention', 'https://arxiv.org/abs/2205.14135'],
  geva: ['Geva và cộng sự (2021), Transformer Feed-Forward Layers Are Key-Value Memories', 'https://arxiv.org/abs/2012.14913'],
  relu: ['Nair & Hinton (2010), Rectified Linear Units Improve Restricted Boltzmann Machines', 'https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf'],
  gelu: ['Hendrycks & Gimpel (2016), Gaussian Error Linear Units (GELUs)', 'https://arxiv.org/abs/1606.08415'],
  layernorm: ['Ba, Kiros & Hinton (2016), Layer Normalization', 'https://arxiv.org/abs/1607.06450'],
  resnet: ['He và cộng sự (2015), Deep Residual Learning for Image Recognition', 'https://arxiv.org/abs/1512.03385'],
  resnet2: ['He và cộng sự (2016), Identity Mappings in Deep Residual Networks', 'https://arxiv.org/abs/1603.05027'],
  bpe: ['Sennrich, Haddow & Birch (2016), Neural Machine Translation of Rare Words with Subword Units', 'https://arxiv.org/abs/1508.07909'],
  press: ['Press & Wolf (2017), Using the Output Embedding to Improve Language Models', 'https://arxiv.org/abs/1608.05859'],
  xiong: ['Xiong và cộng sự (2020), On Layer Normalization in the Transformer Architecture', 'https://arxiv.org/abs/2002.04745'],
  dropout: ['Srivastava và cộng sự (2014), Dropout: A Simple Way to Prevent Neural Networks from Overfitting', 'https://jmlr.org/papers/v15/srivastava14a.html'],
  llama3: ['Llama Team, Meta (2024), The Llama 3 Herd of Models', 'https://arxiv.org/abs/2407.21783'],
  rmsnorm: ['Zhang & Sennrich (2019), Root Mean Square Layer Normalization', 'https://arxiv.org/abs/1910.07467'],
  gqa: ['Ainslie và cộng sự (2023), GQA: Training Generalized Multi-Query Transformer Models', 'https://arxiv.org/abs/2305.13245'],
  mqa: ['Shazeer (2019), Fast Transformer Decoding: One Write-Head is All You Need', 'https://arxiv.org/abs/1911.02150'],
  rope: ['Su và cộng sự (2021), RoFormer: Enhanced Transformer with Rotary Position Embedding', 'https://arxiv.org/abs/2104.09864'],
  glu: ['Shazeer (2020), GLU Variants Improve Transformer', 'https://arxiv.org/abs/2002.05202'],
  topp: ['Holtzman và cộng sự (2019), The Curious Case of Neural Text Degeneration', 'https://arxiv.org/abs/1904.09751'],
  palm: ['Chowdhery và cộng sự (2022), PaLM: Scaling Language Modeling with Pathways', 'https://arxiv.org/abs/2204.02311'],
  vit: ['Dosovitskiy và cộng sự (2020), An Image is Worth 16x16 Words', 'https://arxiv.org/abs/2010.11929'],
  bert: ['Devlin và cộng sự (2018), BERT', 'https://arxiv.org/abs/1810.04805'],
  bn: ['Ioffe & Szegedy (2015), Batch Normalization', 'https://arxiv.org/abs/1502.03167'],
  goyal: ['Goyal và cộng sự (2017), Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour', 'https://arxiv.org/abs/1706.02677'],
  nin: ['Lin, Chen & Yan (2013), Network In Network', 'https://arxiv.org/abs/1312.4400'],
  vgg: ['Simonyan & Zisserman (2014), Very Deep Convolutional Networks (VGG)', 'https://arxiv.org/abs/1409.1556'],
  lstm: ['Hochreiter & Schmidhuber (1997), Long Short-Term Memory', 'https://doi.org/10.1162/neco.1997.9.8.1735'],
  zaremba: ['Zaremba, Sutskever & Vinyals (2014), Recurrent Neural Network Regularization', 'https://arxiv.org/abs/1409.2329'],
  gers: ['Gers, Schmidhuber & Cummins (2000), Learning to Forget: Continual Prediction with LSTM', 'https://doi.org/10.1162/089976600300015015'],
  jozefowicz: ['Jozefowicz, Zaremba & Sutskever (2015), An Empirical Exploration of Recurrent Network Architectures', 'https://proceedings.mlr.press/v37/jozefowicz15.html'],
  colah: ['Olah (2015), Understanding LSTM Networks (bài viết)', 'https://colah.github.io/posts/2015-08-Understanding-LSTMs/'],
  unet: ['Ronneberger, Fischer & Brox (2015), U-Net: Convolutional Networks for Biomedical Image Segmentation', 'https://arxiv.org/abs/1505.04597'],
  odena: ['Odena, Dumoulin & Olah (2016), Deconvolution and Checkerboard Artifacts (Distill)', 'https://distill.pub/2016/deconv-checkerboard/'],
  rumelhart: ['Rumelhart, Hinton & Williams (1986), Learning representations by back-propagating errors', 'https://doi.org/10.1038/323533a0'],
  adam: ['Kingma & Ba (2014), Adam: A Method for Stochastic Optimization', 'https://arxiv.org/abs/1412.6980'],
  heinit: ['He và cộng sự (2015), Delving Deep into Rectifiers (khởi tạo He)', 'https://arxiv.org/abs/1502.01852'],
  momentum: ['Sutskever và cộng sự (2013), On the importance of initialization and momentum in deep learning', 'https://proceedings.mlr.press/v28/sutskever13.html'],
  playground: ['Smilkov & Carter, TensorFlow Playground (ý tưởng gốc của phòng thí nghiệm này)', 'https://playground.tensorflow.org/'],
  adamw: ['Loshchilov & Hutter (2019), Decoupled Weight Decay Regularization', 'https://arxiv.org/abs/1711.05101'],
  dlbook: ['Goodfellow, Bengio & Courville (2016), Deep Learning, chương 7: Regularization', 'https://www.deeplearningbook.org/contents/regularization.html'],
  densenet: ['Huang, Liu, van der Maaten & Weinberger (2017), Densely Connected Convolutional Networks', 'https://arxiv.org/abs/1608.06993'],
  efficientnet: ['Tan & Le (2019), EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks', 'https://arxiv.org/abs/1905.11946'],
  mobilenet: ['Howard và cộng sự (2017), MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications', 'https://arxiv.org/abs/1704.04861'],
  mobilenetv2: ['Sandler và cộng sự (2018), MobileNetV2: Inverted Residuals and Linear Bottlenecks', 'https://arxiv.org/abs/1801.04381'],
  senet: ['Hu, Shen & Sun (2018), Squeeze-and-Excitation Networks', 'https://arxiv.org/abs/1709.01507'],
  swish: ['Ramachandran, Zoph & Le (2017), Searching for Activation Functions', 'https://arxiv.org/abs/1710.05941'],
  stochdepth: ['Huang và cộng sự (2016), Deep Networks with Stochastic Depth', 'https://arxiv.org/abs/1603.09382'],
  vnet: ['Milletari, Navab & Ahmadi (2016), V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation', 'https://arxiv.org/abs/1606.04797'],
  mienye: ['Mienye và cộng sự (2025), Deep Convolutional Neural Networks in Medical Image Analysis: A Review', 'https://doi.org/10.3390/info16030195'],
  swin: ['Liu và cộng sự (2021), Swin Transformer: Hierarchical Vision Transformer using Shifted Windows', 'https://arxiv.org/abs/2103.14030'],
  swinunet: ['Cao và cộng sự (2021), Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation', 'https://arxiv.org/abs/2105.05537'],
  swinunetr: ['Hatamizadeh và cộng sự (2022), Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images', 'https://arxiv.org/abs/2201.01266'],
  ldm: ['Rombach và cộng sự (2022), High-Resolution Image Synthesis with Latent Diffusion Models', 'https://arxiv.org/abs/2112.10752'],
  ddpm: ['Ho, Jain & Abbeel (2020), Denoising Diffusion Probabilistic Models', 'https://arxiv.org/abs/2006.11239'],
  ddim: ['Song, Meng & Ermon (2020), Denoising Diffusion Implicit Models', 'https://arxiv.org/abs/2010.02502'],
  cfg: ['Ho & Salimans (2022), Classifier-Free Diffusion Guidance', 'https://arxiv.org/abs/2207.12598'],
  clip: ['Radford và cộng sự (2021), Learning Transferable Visual Models From Natural Language Supervision (CLIP)', 'https://arxiv.org/abs/2103.00020'],
  vae: ['Kingma & Welling (2013), Auto-Encoding Variational Bayes', 'https://arxiv.org/abs/1312.6114'],
  groupnorm: ['Wu & He (2018), Group Normalization', 'https://arxiv.org/abs/1803.08494'],
  sd15: ['Model card Stable Diffusion v1.5 (Hugging Face)', 'https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5']
};
const ref = (key, note) => ({ t: PAPER[key][0], u: PAPER[key][1], note });

// Ideas shared across models: blocks with these exact names link to each other ("Cũng xuất hiện ở" in the panel).
const CONCEPTS = {
  'Đường tắt residual': ['Cộng residual', 'Cộng đường tắt', 'Add & Norm', 'Cập nhật bộ nhớ cₜ', 'Stochastic depth + cộng'],
  'Cổ chai (bottleneck) và cổ chai ngược': ['Bottleneck (identity)', 'Bottleneck đầu (projection)', 'MBConv (identity)', 'MBConv đầu (đổi shape)', 'Conv 1×1 → 4k = 128'],
  'Attention lõi': ['Scaled Dot-Product Attention'],
  'Multi-head attention': ['Multi-Head Self-Attention', 'Masked Multi-Head Attention', 'Cross-Attention', 'Grouped-Query Attention', 'Window attention (W-MSA)', 'Shifted window attention (SW-MSA)'],
  'Query, key, value': ['Chiếu Q, K, V', 'Q·Kᵀ', 'KV cache'],
  'Mặt nạ': ['Mặt nạ'],
  'Softmax': ['Softmax', 'FC 1000 + Softmax', 'Softmax + Cross-entropy'],
  'Chuẩn hoá': ['LayerNorm', 'LayerNorm cuối', 'RMSNorm', 'RMSNorm cuối', 'BatchNorm', 'GroupNorm'],
  'Hàm kích hoạt': ['ReLU', 'GELU', 'SiLU', 'Sigmoid', 'Quick GELU', 'tanh(cₜ)', 'Kích hoạt 1', 'Kích hoạt 2'],
  'Cổng (gating)': ['Cổng quên f', 'Cổng vào i', 'Cổng ra o', 'Nhân cổng ⊙', 'Quên bớt: f ⊙ cₜ₋₁', 'Ghi mới: i ⊙ g', 'Output hₜ = o ⊙ tanh(cₜ)', 'Squeeze-and-Excitation', 'Nhân theo kênh ⊙'],
  'Khối feed-forward': ['Feed-Forward', 'MLP', 'SwiGLU FFN', 'Feed-Forward GEGLU'],
  'Lớp Linear (kể cả conv 1×1)': ['Linear 2 → H', 'Linear H → H', 'Linear H → 2', 'Linear 512 → 2048', 'Linear 2048 → 512', 'Linear 768 → 3072', 'Linear 3072 → 768', 'Nhánh cổng W_gate', 'Nhánh giá trị W_up', 'Nhánh xuống W_down', 'Chiếu ra Wᴼ', 'Conv 1×1 (nén kênh)', 'Conv 1×1 (mở rộng kênh)', 'Conv 1×1 → 4k = 128', 'Conv 1×1 nén ×0.5', 'Conv 1×1 mở rộng ×6', 'Conv 1×1 chiếu (không kích hoạt)', 'Conv 1×1 → 1280', 'FC giảm (Conv 1×1)', 'FC tăng (Conv 1×1)', 'Linear 96 → 384', 'Linear 384 → 96', 'Linear 384 → 192', 'Linear 768 → 384', 'Linear 1536 → 768', 'Linear 320 → 1280', 'Linear 1280 → 1280'],
  'Embedding': ['Nhúng bước thời gian t', 'Token Embedding', 'Embedding', 'Embedding nguồn + Vị trí', 'Embedding đích + Vị trí', 'Patch Embedding'],
  'Mã hoá vị trí': ['Positional Encoding', 'RoPE', 'Cộng vị trí (PE học được)', 'Bias vị trí tương đối B', 'Sinusoid của t'],
  'Giảm độ phân giải (kim tự tháp)': ['Patch Merging', 'MaxPool 3×3, bước 2', 'MaxPool 2×2, bước 2', 'AvgPool 2×2, bước 2', 'Conv 3×3, bước 2', 'Depthwise 3×3, bước 2', 'Depthwise 5×5, bước 2'],
  'Tích chập': ['Conv 3×3', 'Conv 3×3, bước 2', 'Conv 3×3 (không đệm)', 'Conv 7×7, 64 kênh, bước 2', 'Conv 16×16, bước 16', 'Up-conv 2×2, bước 2', 'Đường tắt: Conv 1×1 + BN', 'Đường tắt: Conv 1×1 bước 2 + BN', 'Conv 1×1 (nén kênh)', 'Conv 1×1 (mở rộng kênh)', 'Conv 1×1 → 2 lớp',
    'Conv 1×1 → 4k = 128', 'Conv 3×3 → k = 32', 'Conv 1×1 nén ×0.5', 'Conv 1×1 mở rộng ×6', 'Depthwise 3×3', 'Depthwise 3×3, bước 2', 'Depthwise 5×5', 'Depthwise 5×5, bước 2', 'Conv 1×1 chiếu (không kích hoạt)', 'Conv 1×1 → 1280', 'Conv 4×4, bước 4', 'Conv 3×3 vào', 'Conv 3×3 ra', 'Phóng to ×2 + Conv 3×3', 'Đường tắt: Conv 1×1'],
  'Phóng to (upsampling)': ['Up-conv 2×2, bước 2', 'Phóng to ×2 + Conv 3×3'],
  'Khung chữ U (nối tắt mã hoá – giải mã)': ['U-Net (2015)', 'U-Net dự đoán nhiễu'],
  'Khối residual tích chập': ['Bottleneck (identity)', 'Bottleneck đầu (projection)', 'ResBlock'],
  'Pooling': ['MaxPool 3×3, bước 2', 'MaxPool 2×2, bước 2', 'AvgPool 2×2, bước 2', 'Global Average Pooling'],
  'Dropout': ['Dropout', 'Cộng + Dropout', 'Stochastic depth + cộng'],
  'Lớp đầu ra': ['Linear → từ điển', 'LM head', 'Head phân loại', 'FC 1000 + Softmax', 'Conv 1×1 → 2 lớp', 'Softmax + Cross-entropy'],
  'Đầu vào': ['Token nguồn', 'Token đích (dịch phải)', 'Token vào', 'Ảnh vào', 'Điểm vào (x, y)', 'Câu mô tả (prompt)', 'Latent nhiễu z_T'],
  'Tách và nối các đầu': ['Tách 8 đầu', 'Tách 12 đầu', 'Nối các đầu', 'Lặp K, V cho nhóm', 'Tách 3 đầu', 'Tách 6 đầu', 'Tách 24 đầu'],
  'Nối (concat)': ['Nối các đầu', 'Nối [hₜ₋₁; xₜ]', 'Cắt và nối (copy & crop)', 'Thêm token [CLS]', 'Nối kênh', 'Gom 2×2 patch', 'Nối skip'],
  'Chồng nhiều lớp giống nhau': ['Encoder', 'Decoder', 'LSTM', 'Bottleneck (identity)', 'MBConv (identity)', 'Khối Swin (W-MSA)', 'Khối Swin (SW-MSA)']
};

const CONV_F = [R`y_{o,i,j} = \sum_{c=1}^{C_{in}} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} W_{o,c,u,v}\; x_{c,\; s i + u - p,\; s j + v - p}`, R`H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1`];

const relu = (shape, size, note = '') => n('ReLU', 'act', shape, {
  what: 'Giữ nguyên số dương, đưa số âm về 0.',
  formula: [R`\mathrm{ReLU}(x) = \max(0, x), \qquad \frac{d}{dx}\mathrm{ReLU}(x) = \begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}`],
  why: 'Không có phi tuyến thì chồng bao nhiêu lớp tuyến tính cũng chỉ bằng một lớp tuyến tính. ReLU rẻ, gradient bằng 1 ở vùng dương nên không bị nhỏ dần như sigmoid, và làm activation thưa (khoảng một nửa bằng 0).' + note,
  example: R`\(\mathrm{ReLU}([-1.5,\ 0.3,\ 2]) = [0,\ 0.3,\ 2]\).`,
  code: 'nn.ReLU(inplace=True)'
}, { size, ref: [ref('relu')] });

const gelu = (shape, size) => n('GELU', 'act', shape, {
  what: R`Nhân \(x\) với xác suất một biến chuẩn nhỏ hơn \(x\): số dương lớn gần như giữ nguyên, số âm lớn về gần 0, quanh 0 thì chuyển mượt.`,
  formula: [R`\mathrm{GELU}(x) = x\,\Phi(x) \approx \tfrac{1}{2}\,x\Big(1 + \tanh\big(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\big)\Big)`],
  why: R`\(\Phi\) là hàm phân phối tích luỹ của phân phối chuẩn. Giống ReLU khi \(|x|\) lớn nhưng mượt quanh 0 và cho qua một chút giá trị âm, nên gradient không bị cắt cụt. BERT, GPT-2 và ViT đều dùng.`,
  example: R`\(\mathrm{GELU}([-1.5,\ 0.3,\ 2]) = [-0.100,\ 0.185,\ 1.954]\). So với ReLU: \([0,\ 0.3,\ 2]\).`,
  code: 'nn.GELU()'
}, { size, ref: [ref('gelu')] });

const silu = (shape, size, note = '') => n('SiLU', 'act', shape, {
  what: 'Nhân x với sigmoid của chính nó. Còn gọi là Swish.',
  formula: [R`\mathrm{SiLU}(x) = x\,\sigma(x) = \frac{x}{1 + e^{-x}}`],
  why: 'Mượt, gần ReLU khi |x| lớn, nhưng cho qua một chút giá trị âm và có đạo hàm khác 0 quanh 0, nên tín hiệu học không bị cắt cụt.' + note,
  example: R`\(\mathrm{SiLU}([-1.5,\ 0.3,\ 2]) = [-0.274,\ 0.172,\ 1.762]\).`,
  code: 'F.silu(x)   # = x * torch.sigmoid(x)'
}, { size, ref: [ref('swish')] });

// CNN pieces shared by ResNet, DenseNet, EfficientNet. tail replaces the last sentence of "why" (pre-activation nets cannot fold BN).
const BN_FOLD = R` Lúc triển khai, BN gộp được hẳn vào conv phía trước: \(W' = \gamma W / \sigma\), \(b' = \beta - \gamma \mu / \sigma\), không tốn thêm phép tính nào.`;
const bn = (C, H, tail = BN_FOLD) => n('BatchNorm', 'norm', `${C}×${H}×${H}`, {
  what: R`Chuẩn hoá từng kênh theo trung bình và phương sai tính trên cả batch và mọi vị trí (\(B \times H \times W\) giá trị), rồi co giãn và dịch bằng \(\gamma, \beta\) học được.`,
  formula: [R`\mu_c = \frac{1}{BHW} \sum_{b,i,j} x_{b,c,i,j}, \qquad \sigma_c^2 = \frac{1}{BHW} \sum_{b,i,j} (x_{b,c,i,j} - \mu_c)^2`, R`y = \gamma_c\, \frac{x - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c`],
  why: R`Giữ phân bố activation ổn định giữa các lớp, nên train được với learning rate lớn và ít nhạy với khởi tạo. Khi train dùng thống kê của batch hiện tại; khi suy luận dùng trung bình trượt đã tích luỹ, nên phải gọi model.eval().` + tail,
  params: R`\(\gamma, \beta\): \(2 \times ${C} = ${fmt(2 * C)}\) tham số học, thêm \(${fmt(2 * C)}\) số thống kê trượt (không học).`,
  code: `nn.BatchNorm2d(${C})`
}, { tag: 'BN', size: [0.14, sp(H), sp(H)], ref: [ref('bn')] });

const gapNode = (C, H, why, more = {}) => n('Global Average Pooling', 'pool', `${C}×${H}×${H} → ${C}`, {
  what: `Lấy trung bình mỗi kênh trên toàn bộ ${H}×${H} vị trí, ra một vector ${C} chiều mô tả cả ảnh.`,
  formula: [R`z_c = \frac{1}{${H} \cdot ${H}} \sum_{i=1}^{${H}} \sum_{j=1}^{${H}} x_{c,i,j}`],
  why,
  code: `z = torch.flatten(nn.AdaptiveAvgPool2d(1)(x), 1)   # (B, ${C})`
}, { tag: 'Avg Pool', size: [ch(C), 0.22, 0.22], ref: [ref('nin', 'global average pooling')], ...more });

// attr = where torchvision keeps the last Linear (model.fc, model.classifier, ...)
const fcOut = (C, attr) => n('FC 1000 + Softmax', 'linear', `${C} → 1000`, {
  what: 'Một lớp Linear cho ra 1000 điểm số, mỗi lớp của ImageNet một điểm, rồi softmax thành xác suất.',
  formula: [R`\text{logits} = W z + b, \qquad W \in \mathbb{R}^{1000 \times ${C}}`, R`p_k = \frac{e^{\text{logit}_k}}{\sum_{j=1}^{1000} e^{\text{logit}_j}}, \qquad \mathcal{L} = -\log p_{y}`],
  why: 'Trong PyTorch, softmax nằm sẵn trong hàm loss (cross_entropy nhận logit), nên model chỉ trả về logit. Muốn dùng mạng cho bài toán khác, thường chỉ cần thay lớp này rồi fine-tune.',
  params: R`\(${C} \times 1000 + 1000 = ${fmt(C * 1000 + 1000)}\).`,
  code: `model.${attr} = nn.Linear(${C}, num_classes)   # thay đầu ra khi fine-tune
loss = F.cross_entropy(model(x), y)`
}, { tag: 'FC → 1000', size: [ch(1000), 0.22, 0.22] });

const layerNorm = (s, d) => n('LayerNorm', 'norm', `${s}×${d}`, {
  what: R`Chuẩn hoá vector ${d} chiều của từng token về trung bình 0, phương sai 1, rồi co giãn và dịch bằng hai vector học được \(\gamma, \beta\).`,
  formula: [R`\mu = \frac{1}{${d}}\sum_{c=1}^{${d}} y_c, \qquad \sigma^2 = \frac{1}{${d}}\sum_{c=1}^{${d}} (y_c - \mu)^2`, R`\mathrm{LN}(y)_c = \gamma_c\, \frac{y_c - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_c`],
  why: 'Khác BatchNorm, thống kê được tính trên từng token riêng, không phụ thuộc vào batch hay độ dài chuỗi, nên lúc train và lúc suy luận hành xử giống hệt nhau. Đó là lý do mô hình dạng Transformer dùng LayerNorm còn CNN hay dùng BatchNorm.',
  example: R`\(y = [1, 2, 3, 6]\): \(\mu = 3\), \(\sigma^2 = 3.5\), \(\sigma = 1.871\) → \([-1.069,\ -0.535,\ 0,\ 1.604]\) (với \(\gamma = 1, \beta = 0\)).`,
  params: R`\(\gamma, \beta\): \(2 \times ${d} = ${fmt(2 * d)}\).`,
  code: `norm = nn.LayerNorm(${d})`
}, { size: SLAB, ref: [ref('layernorm'), ref('bn', 'để so sánh với BatchNorm')] });

const addLeaf = (s, d) => n('Cộng residual', 'add', `${s}×${d}`, {
  what: R`Cộng từng phần tử: đầu vào \(x\) của sublayer cộng với output của nó.`,
  formula: [R`y = x + F(x)`, R`\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}`],
  why: R`Số hạng \(I\) bảo đảm gradient luôn có một đường về các lớp đầu mà không bị nhân qua ma trận trọng số nào, nên không bị triệt tiêu dù mạng sâu.`,
  code: `y = x + h`
}, { tag: 'Cộng', size: BALL, exp: 'res', ref: [ref('resnet', 'mục 3.1–3.2'), ref('resnet2', 'phân tích gradient qua đường tắt')] }); // exp = interactive experiment in the panel (index.html)

function sdpa(kind, q, kv, c) {
  const masked = kind === 'masked', { h, dk } = c;
  const kids = [
    n('Q·Kᵀ', 'attn', `→ ${h}×${q}×${kv}`, {
      what: R`Tích vô hướng giữa mọi cặp query–key cho ra ma trận điểm \(${q} \times ${kv}\): ô \((i, j)\) đo token \(i\) muốn chú ý tới token \(j\) đến mức nào.`,
      formula: [R`S = Q K^\top, \qquad S_{ij} = q_i \cdot k_j = \sum_{c=1}^{${dk}} q_{ic}\, k_{jc}`],
      why: R`Tích vô hướng lớn khi hai vector cùng hướng. Vì \(W^Q\) và \(W^K\) được học, mạng tự quyết định "cùng hướng" nghĩa là gì cho từng đầu.`,
      example: R`\(q = [1, 0, 1]\): \(q \cdot k_1 = 1 \cdot 1 + 0 \cdot 1 + 1 \cdot 0 = 1\), \(q \cdot k_2 = 0 + 0 + 1 \cdot 2 = 2\). Hàng điểm: \([1,\ 2]\).`,
      params: R`Không có tham số. Chi phí \(${h} \cdot ${q} \cdot ${kv} \cdot ${dk}\) phép nhân-cộng.`,
      code: `scores = q @ k.transpose(-2, -1)   # (B, ${h}, ${q}, ${kv})`
    }, { size: [1.2, 1.3, 1.3] }),
    n('Chia √dₖ', 'norm', `${h}×${q}×${kv}`, {
      what: R`Chia mọi điểm cho \(${sqrtTeX(dk)}\).`,
      formula: [R`\tilde S = \frac{S}{\sqrt{d_k}}`, R`\mathrm{Var}(q \cdot k) = \sum_{c=1}^{d_k} \mathrm{Var}(q_c\, k_c) = d_k`],
      why: R`Giả sử các thành phần của \(q, k\) độc lập, trung bình 0, phương sai 1. Khi đó \(q \cdot k\) có phương sai \(d_k = ${dk}\), tức độ lệch chuẩn \(${sqrtTeX(dk)}\). Softmax trên các số chênh nhau cỡ đó gần như thành one-hot, gradient qua nó gần bằng 0 và mạng học rất chậm. Chia \(\sqrt{d_k}\) đưa phương sai về 1.`,
      example: R`\([1,\ 2] / \sqrt{3} = [0.577,\ 1.155]\).`,
      code: `scores = scores / math.sqrt(${dk})`
    }, { tag: '÷ √dₖ', size: SLAB, ref: [ref('vaswani', 'mục 3.2.1, chú thích 4')] }),
    ...(c.noMask ? [] : [n('Mặt nạ', 'norm', `${h}×${q}×${kv}`, masked ? {
      what: R`Cộng \(-\infty\) vào các ô \((i, j)\) có \(j > i\), tức các token tương lai. Sau softmax, các ô này thành đúng 0.`,
      formula: [R`M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}`],
      why: R`Vì \(e^{-\infty} = 0\), token \(i\) không nhận được chút thông tin nào từ phía sau, và ma trận trọng số thành tam giác dưới. Nhờ vậy cả chuỗi chạy song song khi train mà kết quả vẫn đúng như khi sinh từng token.`,
      example: R`Với 3 token: \(M = \begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{bmatrix}\).`,
      code: `causal = torch.triu(torch.ones(${q}, ${q}, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(causal, float("-inf"))`
    } : {
      what: R`Cộng \(-\infty\) vào các cột ứng với token đệm (padding) để không token nào chú ý vào chúng. Encoder và cross-attention chỉ cần mặt nạ này.`,
      formula: [R`M_{ij} = \begin{cases} 0 & j \notin \text{pad} \\ -\infty & j \in \text{pad} \end{cases}`],
      why: 'Các câu trong một batch dài ngắn khác nhau nên phải đệm cho bằng nhau. Mặt nạ bảo đảm phần đệm không ảnh hưởng tới kết quả.',
      code: `scores = scores.masked_fill(pad[:, None, None, :], float("-inf"))`
    }, { size: SLAB })]),
    n('Softmax', 'act', `${h}×${q}×${kv}`, {
      what: 'Chuẩn hoá từng hàng thành phân bố xác suất: mọi trọng số không âm và cộng lại bằng 1.',
      formula: [R`A_{ij} = \frac{\exp(\tilde S_{ij})}{\sum_{k} \exp(\tilde S_{ik})}`],
      why: 'Cài đặt thật trừ đi giá trị lớn nhất của hàng trước khi lấy mũ để khỏi tràn số. Kết quả không đổi vì softmax không đổi khi cộng cùng một hằng số vào mọi phần tử.' + (c.drop ? ` Sau softmax còn có dropout ${c.drop} trên chính các trọng số này.` : ''),
      example: R`\(e^{0.577} = 1.781\), \(e^{1.155} = 3.173\), tổng \(4.954\) → \([0.360,\ 0.640]\).`,
      code: `attn = scores.softmax(dim=-1)` + (c.drop ? `\nattn = F.dropout(attn, p=${c.drop}, training=self.training)` : '')
    }, { size: SLAB }),
    n('A·V', 'attn', `→ ${h}×${q}×${dk}`, {
      what: 'Mỗi token nhận về một tổ hợp lồi của các value, theo đúng trọng số vừa tính.',
      formula: [R`O = A V, \qquad o_i = \sum_{j} A_{ij}\, v_j`],
      why: R`Vì trọng số không âm và tổng bằng 1, \(o_i\) luôn nằm trong bao lồi của các value: attention chỉ pha trộn thông tin có sẵn. Biến đổi mới do \(W^O\) và FFN đảm nhận.`,
      example: R`\(0.360 \cdot [1, 0] + 0.640 \cdot [0, 1] = [0.360,\ 0.640]\).`,
      code: `out = attn @ v   # (B, ${h}, ${q}, ${dk})`
    }, { size: [1.2, 1.5, 1.3] })
  ];
  // live = which matrix of the toy run (live.js) this step shows; off for images (ViT) and cross-attention
  if (!c.noLive) ['qk', 'scale', ...(c.noMask ? [] : ['mask']), 'softmax', 'av'].forEach((k, j) => kids[j].live = k);
  const drop = c.drop ? `, dropout_p=${c.drop}` : '';
  return n('Scaled Dot-Product Attention', 'attn', `${h}×${q}×${dk} → ${h}×${q}×${dk}`, {
    what: 'Phép tính lõi của attention, chạy độc lập trên từng đầu. Mỗi query được so với mọi key, điểm số được đổi thành trọng số, rồi các value được lấy trung bình theo trọng số đó.',
    formula: [R`\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}${c.noMask ? '' : ' + M'}\right) V, \qquad d_k = ${dk}`],
    why: R`Đây là một phép tra cứu mềm và khả vi: thay vì lấy đúng một value có key khớp nhất, ta lấy pha trộn của tất cả theo mức độ khớp. Ma trận điểm cỡ \(${q} \times ${kv}\) là lý do chi phí tăng theo bình phương độ dài chuỗi. FlashAttention tính đúng công thức này nhưng theo từng khối nhỏ, không bao giờ lưu cả ma trận đó.`,
    example: R`Một token, \(d_k = 3\) cho dễ: \(q = [1, 0, 1]\), hai key \(k_1 = [1, 1, 0]\), \(k_2 = [0, 0, 2]\), hai value \(v_1 = [1, 0]\), \(v_2 = [0, 1]\). Điểm \([1,\ 2]\) → chia \(\sqrt 3\) → \([0.577,\ 1.155]\) → softmax \([0.360,\ 0.640]\) → output \([0.360,\ 0.640]\). Mở từng bước bên trong để xem chi tiết.`,
    code: masked ? `out = F.scaled_dot_product_attention(q, k, v, is_causal=True${drop})   # (B, ${h}, ${q}, ${dk})`
      : c.noMask ? `out = F.scaled_dot_product_attention(q, k, v)   # (B, ${h}, ${q}, ${dk})`
      : `out = F.scaled_dot_product_attention(q, k, v, attn_mask=pad_mask${drop})   # (B, ${h}, ${q}, ${dk})`
  }, { tag: 'Attention lõi', repeat: h, hint: `${kids.length} bước · ${h} đầu`, size: [1.5, 1.6, 1.3], children: kids, gap: 1.5, // room for the live heatmaps
    ref: [ref('vaswani', 'mục 3.2.1'), ref('flash')],
    ...(c.noLive ? {} : { live: 'sdpa', liveCausal: masked }) });
}

// Classic multi-head attention with biases (Transformer, ViT). LLaMA builds its own (GQA + RoPE).
function mha(kind, q, kv, c) {
  const cross = kind === 'cross', { d, h, dk } = c;
  const names = { self: ['Multi-Head Self-Attention', 'Self-Attention'], masked: ['Masked Multi-Head Attention', 'Masked Attention'], cross: ['Cross-Attention', 'Cross-Attention'] };
  const drop = c.drop ? `, dropout=${c.drop}` : '';
  return n(names[kind][0], 'attn', cross ? `Q ${q}×${d} · K,V ${kv}×${d} → ${q}×${d}` : `${q}×${d} → ${q}×${d}`, {
    what: {
      self: `Mỗi token tạo ra một query, một key và một value từ chính chuỗi đầu vào. Output của một token là trung bình có trọng số các value của mọi token, với trọng số đo mức khớp giữa query của nó và key của từng token. Có ${h} "đầu" chạy song song, mỗi đầu trong một không gian con ${dk} chiều.`,
      masked: R`Giống self-attention, nhưng token ở vị trí \(t\) chỉ được nhìn các vị trí từ 1 đến \(t\). Nhờ mặt nạ này, khi train cả câu đích chạy song song mà không token nào "nhìn trộm" đáp án phía sau.`,
      cross: R`Query lấy từ decoder, còn key và value lấy từ output \(H\) của encoder. Đây là chỗ decoder tra cứu câu nguồn: mỗi token đích tự chọn nên chú ý vào những từ nguồn nào.`
    }[kind],
    formula: [
      R`\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_{${h}})\, W^O`,
      R`\mathrm{head}_i = \mathrm{Attention}(Q W_i^Q,\ K W_i^K,\ V W_i^V)`,
      { self: R`Q = K = V = X`, masked: R`Q = K = V = Y, \qquad M_{ij} = -\infty \ \text{khi}\ j > i`, cross: R`Q = A_{\text{dec}}, \qquad K = V = H_{\text{enc}}` }[kind]
    ],
    why: `Một đầu chỉ tạo được một phân bố chú ý cho mỗi token, tức là một kiểu quan hệ. Nhiều đầu cho phép theo dõi nhiều kiểu cùng lúc: đầu này nối động từ với chủ ngữ, đầu kia nối đại từ với danh từ nó thay thế. Chia ${d} = ${h} × ${dk} nên tổng chi phí xấp xỉ một đầu ${d} chiều.`
      + (cross ? ' Trong dịch máy, các đầu cross-attention thường học ra gần đúng bảng gióng hàng (alignment) giữa từ nguồn và từ đích.' : ''),
    params: R`4 ma trận \(${d} \times ${d}\) (\(W^Q, W^K, W^V, W^O\)) cộng bias: \(${fmt(4 * d * d + 4 * d)}\).`,
    code: {
      self: `mha = nn.MultiheadAttention(${d}, ${h}${drop}, batch_first=True)\n` + (c.noMask ? `out, w = mha(x, x, x)   # w: (B, ${q}, ${q})` : `out, w = mha(x, x, x, key_padding_mask=pad)   # w: (B, ${q}, ${q})`),
      masked: `causal = torch.triu(torch.ones(${q}, ${q}, dtype=torch.bool), diagonal=1)
out, w = mha(y, y, y, attn_mask=causal)        # w: (B, ${q}, ${q})`,
      cross: `out, w = mha(a, H, H, key_padding_mask=src_pad)   # w: (B, ${q}, ${kv})`
    }[kind]
  }, { tag: names[kind][1], size: [1.6, 1.8, 1.5], ref: [ref('vaswani', kind === 'self' ? 'mục 3.2.2' : 'mục 3.2.2–3.2.3')],
    extra: cross ? [{ a: { ext: 0, s: 'b', d: 1.4 }, b: { i: 0, s: 'b' }, kind: 'cross', label: 'H (encoder)' }] : [],
    children: [
      n('Chiếu Q, K, V', 'linear', cross ? `Q ${q}×${d} · K,V ${kv}×${d}` : `${q}×${d} → 3 × ${q}×${d}`, {
        what: 'Ba phép biến đổi tuyến tính riêng tạo ra ba vai trò cho mỗi token: query (tôi đang tìm gì), key (tôi có gì để được tìm thấy), value (tôi gửi đi nội dung gì nếu được chọn).',
        formula: [cross ? R`Q = A_{\text{dec}} W^Q, \quad K = H_{\text{enc}} W^K, \quad V = H_{\text{enc}} W^V` : R`Q = X W^Q, \quad K = X W^K, \quad V = X W^V`, R`W^Q, W^K, W^V \in \mathbb{R}^{${d} \times ${d}}`],
        why: 'Tách key và value ra hai ma trận là điểm mấu chốt: thứ dùng để được tìm thấy không nhất thiết là thứ được gửi đi. Giống tra từ điển: tìm theo mục từ (key), đọc phần giải nghĩa (value).',
        params: R`\(3 \times (${d} \cdot ${d} + ${d}) = ${fmt(3 * (d * d + d))}\).`,
        code: `q, k, v = self.wq(x), self.wk(x), self.wv(x)   # mỗi cái là nn.Linear(${d}, ${d})`
      }, { tag: 'Q, K, V', size: [1.2, 1.5, 1.3] }),
      n(`Tách ${h} đầu`, 'shape', `${q}×${d} → ${h}×${q}×${dk}`, {
        what: `Không tính toán gì, chỉ đổi cách nhìn tensor: ${d} chiều được cắt thành ${h} đoạn ${dk} chiều, mỗi đoạn là input của một đầu.`,
        formula: [R`\mathbb{R}^{${q} \times ${d}} \xrightarrow{\text{view}} \mathbb{R}^{${q} \times ${h} \times ${dk}} \xrightarrow{\text{transpose}} \mathbb{R}^{${h} \times ${q} \times ${dk}}`],
        why: R`Vì chỉ là reshape, ma trận \(W^Q\) cỡ \(${d} \times ${d}\) tương đương ${h} ma trận \(${d} \times ${dk}\) đặt cạnh nhau: mỗi đầu có phép chiếu riêng mà không tốn thêm tham số nào.`,
        code: `q = q.view(B, ${q}, ${h}, ${dk}).transpose(1, 2)   # (B, ${h}, ${q}, ${dk})`
      }, { tag: 'Tách đầu', size: [0.6, 1.5, 1.3] }),
      sdpa(kind, q, kv, cross ? { ...c, noLive: true } : c),
      n('Nối các đầu', 'shape', `${h}×${q}×${dk} → ${q}×${d}`, {
        what: `Đặt output của ${h} đầu cạnh nhau để trở lại ${d} chiều. Ngược lại với bước tách đầu.`,
        formula: [R`\mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_{${h}}) \in \mathbb{R}^{${q} \times ${d}}`],
        code: `out = out.transpose(1, 2).reshape(B, ${q}, ${d})`
      }, { tag: 'Nối đầu', size: [0.6, 1.5, 1.3] }),
      n('Chiếu ra Wᴼ', 'linear', `${q}×${d} → ${q}×${d}`, {
        what: 'Một lớp Linear trộn thông tin giữa các đầu.',
        formula: [R`\mathrm{MHA}(X) = \mathrm{Concat}(\dots)\, W^O + b^O, \qquad W^O \in \mathbb{R}^{${d} \times ${d}}`],
        why: R`Không có bước này, mỗi đầu chỉ ghi được vào đúng ${dk} chiều riêng của nó. \(W^O\) cho phép kết quả của mọi đầu được ghi vào bất kỳ hướng nào trong ${d} chiều.`,
        params: R`\(${d} \cdot ${d} + ${d} = ${fmt(d * d + d)}\).`,
        code: `out = self.wo(out)   # nn.Linear(${d}, ${d})`
      }, { tag: 'Wᴼ', size: [1.2, 1.5, 1.3] })
    ] });
}

// Two-layer position-wise MLP with biases (Transformer: ReLU, ViT: GELU). LLaMA uses SwiGLU instead.
function ffn(s, c) {
  const { d, dff } = c, g = c.act === 'gelu', A = g ? 'GELU' : 'ReLU';
  const p = 2 * d * dff + dff + d;
  return n(g ? 'MLP' : 'Feed-Forward', 'linear', `${s}×${d} → ${s}×${d}`, {
    what: 'MLP hai lớp áp dụng riêng cho từng token, cùng một bộ trọng số ở mọi vị trí. Attention trộn thông tin giữa các token; khối này xử lý thông tin bên trong từng token.',
    formula: [g ? R`\mathrm{MLP}(x) = \mathrm{GELU}(x W_1 + b_1)\, W_2 + b_2` : R`\mathrm{FFN}(x) = \max(0,\ x W_1 + b_1)\, W_2 + b_2`, R`W_1 \in \mathbb{R}^{${d} \times ${dff}}, \qquad W_2 \in \mathbb{R}^{${dff} \times ${d}}`],
    why: R`Chiếm khoảng 2/3 tham số của mỗi lớp. Một cách hiểu phổ biến: khối này là bộ nhớ key–value. Mỗi cột của \(W_1\) nhận ra một mẫu, hàng tương ứng của \(W_2\) là thông tin được ghi vào dòng residual khi mẫu đó xuất hiện.`,
    params: R`\(2 \cdot ${d} \cdot ${dff} + ${dff} + ${d} = ${fmt(p)}\).`,
    code: `ffn = nn.Sequential(nn.Linear(${d}, ${dff}), nn.${A}(),${c.drop ? ` nn.Dropout(${c.drop}),` : ''}
                    nn.Linear(${dff}, ${d}))`
  }, { tag: g ? 'MLP' : 'Feed-Forward', size: [1.4, 1.7, 1.4], ref: [ref('vaswani', 'mục 3.3'), ref('geva')], children: [
    n(`Linear ${d} → ${dff}`, 'linear', `${s}×${d} → ${s}×${dff}`, {
      what: `Mở rộng mỗi token từ ${d} lên ${dff} chiều.`,
      formula: [R`h = x W_1 + b_1`],
      why: `Không gian rộng gấp ${dff / d} cho phi tuyến phía sau nhiều chỗ để tách các đặc trưng ra, trước khi nén lại.`,
      params: R`\(${d} \cdot ${dff} + ${dff} = ${fmt(d * dff + dff)}\).`,
      code: `h = self.w1(x)   # nn.Linear(${d}, ${dff})`
    }, { tag: 'Linear ↑', size: [1.7, 1.7, 1.4] }),
    g ? gelu(`${s}×${dff}`, SLAB) : relu(`${s}×${dff}`, SLAB, ' Các mô hình mới thường dùng GELU hoặc SwiGLU thay cho ReLU.'),
    n(`Linear ${dff} → ${d}`, 'linear', `${s}×${dff} → ${s}×${d}`, {
      what: `Nén về lại ${d} chiều để cộng được vào residual.`,
      formula: [R`y = \mathrm{${A}}(h)\, W_2 + b_2`],
      params: R`\(${dff} \cdot ${d} + ${d} = ${fmt(dff * d + d)}\).`,
      code: `y = self.w2(${g ? 'F.gelu' : 'F.relu'}(h))   # nn.Linear(${dff}, ${d})`
    }, { tag: 'Linear ↓', size: [1.0, 1.5, 1.3] })
  ] });
}

const imageIn = n('Ảnh vào', 'io', '3×224×224', {
  what: 'Ảnh RGB đã đổi về 224×224 và chuẩn hoá theo trung bình, độ lệch chuẩn của từng kênh trên ImageNet.',
  formula: [R`\hat{x}_c = \frac{x_c - \mu_c}{\sigma_c}, \qquad \mu = (0.485,\ 0.456,\ 0.406), \quad \sigma = (0.229,\ 0.224,\ 0.225)`],
  why: 'Chuẩn hoá đưa input về quanh 0 với phương sai cỡ 1, giống phân bố mạng đã thấy khi train. Quên bước này khi dùng mô hình train sẵn là lỗi rất hay gặp: mạng vẫn chạy nhưng dự đoán kém hẳn.',
  code: `tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
x = tf(img).unsqueeze(0)   # (1, 3, 224, 224)`
}, { tag: 'Ảnh', size: [0.06, sp(224), sp(224)] });
