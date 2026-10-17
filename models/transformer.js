// Transformer base (Vaswani et al., 2017). Post-LN encoder-decoder for translation.
(() => {
const TF = { d: 512, h: 8, dk: 64, dff: 2048, act: 'relu', drop: 0.1 };

const tokSrc = n('Token nguồn', 'io', 'n', {
  what: 'Câu nguồn đã được tokenizer cắt thành các mảnh từ con (subword, BPE) rồi đổi thành số nguyên. Mô hình không bao giờ thấy chữ, chỉ thấy các id này.',
  why: 'Subword là điểm cân bằng giữa ký tự (chuỗi quá dài) và nguyên từ (từ điển quá lớn, gặp từ lạ là chịu). Bài gốc dùng một bộ BPE chung cho tiếng Anh và tiếng Đức, khoảng 37 000 token.',
  example: '"I love cats" → ["I", "▁love", "▁cats"] → [40, 1842, 11875] (id minh hoạ).',
  code: `ids = tokenizer("I love cats", return_tensors="pt").input_ids   # (B, n)`
}, { tag: 'Token nguồn', size: [0.5, 1.4, 0.5], ref: [ref('bpe')] });

const tokTgt = n('Token đích (dịch phải)', 'io', 'm', {
  what: R`Câu đích được dịch phải một vị trí bằng cách thêm token \(\langle bos \rangle\) vào đầu. Tại vị trí \(t\), decoder nhận \(y_{\lt t}\) và phải đoán \(y_t\).`,
  why: 'Khi train, cả câu đích có sẵn nên được đưa vào một lần (teacher forcing) và loss được tính ở mọi vị trí song song. Khi suy luận, chuỗi này được sinh dần: mỗi token vừa đoán lại được nối vào input.',
  example: 'Đích "Ich liebe Katzen" → input decoder [⟨bos⟩, Ich, liebe, Katzen], nhãn cần đoán [Ich, liebe, Katzen, ⟨eos⟩].',
  code: `dec_in = torch.cat([bos, tgt[:, :-1]], dim=1)   # dịch phải
labels = tgt`
}, { tag: 'Token đích', size: [0.5, 1.4, 0.5] });

const embed = (s, tgt) => n(tgt ? 'Embedding đích + Vị trí' : 'Embedding nguồn + Vị trí', 'embed', `${s} → ${s}×512`, {
  what: 'Biến mỗi id thành một vector 512 chiều mang nghĩa, rồi cộng thêm một vector mã hoá vị trí của token trong câu.',
  formula: [R`h^{(0)}_i = \mathrm{Dropout}\big(E[x_i]\sqrt{d_{\text{model}}} + PE_i\big)`],
  why: 'Bài gốc dùng chung một ma trận E cho embedding nguồn, embedding đích và lớp Linear cuối (weight tying). Cách này tiết kiệm khoảng 19M tham số và giúp token hiếm học tốt hơn.',
  code: `x = self.drop(self.emb(ids) * math.sqrt(512) + self.pe[:n])   # (B, ${s}, 512)`
}, { tag: 'Embedding + PE', size: [1.3, 1.6, 1.4], ref: [ref('vaswani', 'mục 3.4'), ref('press', 'dùng chung trọng số')], children: [
  n('Token Embedding', 'embed', `${s} → ${s}×512`, {
    what: R`Tra bảng: id \(x_i\) lấy ra hàng thứ \(x_i\) của ma trận \(E\), rồi nhân với \(\sqrt{512} \approx 22.6\).`,
    formula: [R`e_i = E[x_i]\,\sqrt{d_{\text{model}}}, \qquad E \in \mathbb{R}^{V \times 512}`, R`E[x_i] = \mathrm{onehot}(x_i)^\top E`],
    why: R`Tra bảng cho kết quả giống hệt nhân vector one-hot với \(E\) nhưng không tốn phép nhân nào. \(E\) được học cùng cả mạng, nên các token được dùng giống nhau sẽ có vector gần nhau. Nhân \(\sqrt{d_{\text{model}}}\) để embedding (khởi tạo nhỏ) có độ lớn tương xứng với PE, vốn có biên độ 1.`,
    params: R`\(V \times 512 \approx 37\,000 \times 512 \approx 18.9\)M, dùng chung với lớp Linear cuối.`,
    code: `emb = nn.Embedding(V, 512)
e = emb(ids) * math.sqrt(512)   # (B, ${s}, 512)`
  }, { tag: 'Embedding', size: [1.2, 1.5, 1.3] }),
  n('Positional Encoding', 'embed', `${s}×512`, {
    what: 'Attention tự nó không biết thứ tự: đảo các token thì output chỉ bị đảo theo. PE gắn cho mỗi vị trí một "dấu vân tay" sin/cos riêng để mô hình biết token đứng thứ mấy.',
    formula: [R`PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/512}}\right)`, R`PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/512}}\right)`],
    why: R`Mỗi cặp chiều \((2i, 2i{+}1)\) là một kim đồng hồ quay với tần số riêng, bước sóng từ \(2\pi\) đến \(10000 \cdot 2\pi\). Với độ lệch \(k\) cố định, \(PE_{pos+k}\) là một phép quay của \(PE_{pos}\) (ma trận quay chỉ phụ thuộc \(k\)), nên mô hình dễ học kiểu chú ý "token cách mình \(k\) vị trí". RoPE trong các LLM hiện nay phát triển trực tiếp từ ý này (xem LLaMA 3).`,
    example: R`\(pos = 1,\ i = 0\): \([\sin 1,\ \cos 1] = [0.841,\ 0.540]\). \(pos = 1,\ i = 255\) (cặp chiều cuối): tần số \(10000^{-510/512} \approx 1.04 \times 10^{-4}\), nên được \([0.0001,\ 1.0000]\). Chiều thấp đổi nhanh theo vị trí, chiều cao gần như đứng yên.`,
    params: 'Không có tham số. Bài gốc thử cả PE học được, kết quả gần như nhau.',
    code: `pos = torch.arange(n)[:, None]          # (n, 1)
i = torch.arange(0, 512, 2)             # các chỉ số 2i
ang = pos / 10000 ** (i / 512)          # (n, 256)
pe = torch.zeros(n, 512)
pe[:, 0::2] = torch.sin(ang)
pe[:, 1::2] = torch.cos(ang)`
  }, { tag: 'Vị trí (PE)', size: [0.9, 1.5, 1.3], ref: [ref('vaswani', 'mục 3.5')] }),
  n('Cộng + Dropout', 'add', `${s}×512`, {
    what: 'Cộng embedding với PE theo từng phần tử, rồi dropout.',
    formula: [R`h^{(0)} = \mathrm{Dropout}_{p = 0.1}(e + PE)`],
    why: 'Cộng thay vì nối (concat) để giữ nguyên 512 chiều. Trong không gian nhiều chiều, hai loại thông tin này vẫn tách được gần như trực giao, nên cộng không làm mất mát đáng kể.',
    code: `h0 = F.dropout(e + pe, p=0.1, training=self.training)`
  }, { tag: 'Cộng', size: BALL })
] });

const addNorm = s => n('Add & Norm', 'norm', `${s}×512`, {
  what: 'Bọc quanh mỗi sublayer: cộng lại đầu vào ban đầu của sublayer (residual), rồi chuẩn hoá từng token.',
  formula: [R`\mathrm{AddNorm}(x) = \mathrm{LN}\big(x + \mathrm{Dropout}(\mathrm{Sublayer}(x))\big)`],
  why: 'Residual tạo một đường đi thẳng cho gradient xuyên qua mọi lớp, nên sublayer chỉ cần học phần hiệu chỉnh cộng thêm. LayerNorm giữ độ lớn các vector ổn định từ lớp này sang lớp khác. Đây cũng chính là đường tắt của ResNet.',
  code: `x = self.norm(x + self.drop(sublayer(x)))       # Post-LN như bài gốc
# Pre-LN (GPT, LLaMA, ViT): x = x + self.drop(sublayer(self.norm(x)))`
}, { size: [0.42, 1.8, 1.5], hint: '3 bước', ref: [ref('vaswani', 'mục 3.1'), ref('xiong', 'Post-LN và Pre-LN')],
  extra: [{ a: { ext: 1, s: 't', d: 1.1 }, b: { i: 1, s: 't' }, kind: 'short', label: 'x' }],
  children: [
    n('Dropout', 'act', `${s}×512`, {
      what: 'Khi train, tắt ngẫu nhiên 10% phần tử trong output của sublayer.',
      formula: [R`\mathrm{Dropout}(h)_c = \frac{m_c\, h_c}{1 - p}, \qquad m_c \sim \mathrm{Bernoulli}(1 - p), \quad p = 0.1`],
      why: R`Chia cho \(1 - p\) để giữ nguyên kỳ vọng, nhờ vậy lúc suy luận chỉ cần bỏ dropout đi. Mạng không thể dựa dẫm vào vài neuron cụ thể nên tổng quát tốt hơn.`,
      example: R`\(h = [2, 4, 6, 8]\), mặt nạ \([1, 0, 1, 1]\), \(p = 0.25\) → \([2.67,\ 0,\ 8,\ 10.67]\).`,
      code: `h = F.dropout(h, p=0.1, training=self.training)`
    }, { size: SLAB, ref: [ref('dropout')] }),
    addLeaf(s, 512),
    layerNorm(s, 512)
  ] });

const ENCODER = n('Encoder', 'block', 'n×512 → n×512', {
  what: '6 lớp giống hệt nhau về cấu trúc (trọng số riêng) xếp chồng. Mỗi lớp có 2 sublayer: self-attention để các token trao đổi thông tin với nhau, và feed-forward để xử lý riêng từng token. Mỗi sublayer được bọc bởi residual và LayerNorm.',
  formula: [R`z = \mathrm{LN}\big(x + \mathrm{MHA}(x, x, x)\big)`, R`x' = \mathrm{LN}\big(z + \mathrm{FFN}(z)\big)`],
  why: 'Shape giữ nguyên qua mọi lớp nên chồng bao nhiêu lớp cũng được. Có thể hình dung mỗi token chạy dọc một "dòng residual": mỗi sublayer đọc dòng đó rồi cộng thêm một ít thông tin. Bài gốc đặt LayerNorm sau phép cộng (Post-LN). GPT, LLaMA, ViT và hầu hết mô hình hiện nay đặt trước (Pre-LN) vì mạng sâu train ổn định hơn.',
  params: R`Mỗi lớp \(3\,152\,384\): MHA \(1\,050\,624\), FFN \(2\,099\,712\), 2 LayerNorm \(2048\). Cả 6 lớp khoảng 18.9M.`,
  code: `layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
                                   dropout=0.1, batch_first=True)
encoder = nn.TransformerEncoder(layer, num_layers=6)
H = encoder(x, src_key_padding_mask=pad)   # (B, n, 512)`
}, { repeat: 6, hint: '4 phần · 6 lớp', size: [2.0, 2.2, 1.6], ref: [ref('vaswani', 'mục 3.1')],
  children: [mha('self', 'n', 'n', TF), addNorm('n'), ffn('n', TF), addNorm('n')],
  skips: [[0, 1, 'x'], [2, 3, 'z']] });

const DECODER = n('Decoder', 'block', 'm×512 → m×512', {
  what: 'Giống encoder nhưng mỗi lớp có 3 sublayer: masked self-attention (token chỉ nhìn các token đứng trước), cross-attention (nhìn sang output của encoder), và feed-forward.',
  formula: [R`a = \mathrm{LN}\big(y + \mathrm{MaskedMHA}(y)\big)`, R`c = \mathrm{LN}\big(a + \mathrm{MHA}(a, H, H)\big)`, R`y' = \mathrm{LN}\big(c + \mathrm{FFN}(c)\big)`],
  why: R`Output \(H\) của encoder được tính một lần, mọi lớp decoder đều dùng lại. Khi suy luận, decoder chạy lặp: mỗi bước sinh một token, nối vào input rồi chạy tiếp. Để khỏi tính lại K, V của các token cũ ở mỗi bước, các hệ thống thật lưu chúng lại (KV cache, xem LLaMA 3).`,
  params: R`Mỗi lớp \(4\,204\,032\), hơn encoder một khối attention và một LayerNorm. Cả 6 lớp khoảng 25.2M.`,
  code: `layer = nn.TransformerDecoderLayer(512, 8, 2048, dropout=0.1, batch_first=True)
decoder = nn.TransformerDecoder(layer, num_layers=6)
mask = nn.Transformer.generate_square_subsequent_mask(m)
out = decoder(y, H, tgt_mask=mask)   # (B, m, 512)`
}, { repeat: 6, hint: '6 phần · 6 lớp', size: [2.0, 2.2, 1.6], ref: [ref('vaswani', 'mục 3.1')],
  children: [mha('masked', 'm', 'm', TF), addNorm('m'), mha('cross', 'm', 'n', TF), addNorm('m'), ffn('m', TF), addNorm('m')],
  skips: [[0, 1, 'y'], [2, 3, 'a'], [4, 5, 'c']],
  extra: [{ a: { ext: 2, s: 'b', d: 1.5 }, b: { i: 2, s: 'b' }, kind: 'cross', label: 'H' }] });

MODELS.transformer = n('Transformer', 'block', 'token nguồn n, token đích m → xác suất m×V', {
  what: 'Mô hình dịch máy (seq2seq) chỉ dùng attention, bỏ hẳn RNN và CNN. Encoder đọc câu nguồn thành các vector ngữ cảnh. Decoder sinh câu đích từng token, mỗi bước tra cứu lại câu nguồn qua cross-attention.',
  formula: [R`P(y \mid x) = \prod_{t=1}^{m} P\big(y_t \mid y_{\lt t},\ \mathrm{Encoder}(x)\big)`],
  why: R`Với RNN, thông tin từ token đầu muốn tới token thứ \(n\) phải đi qua \(n\) bước tuần tự. Với attention, mọi cặp token nối nhau trực tiếp trong 1 bước, và cả chuỗi được tính song song trên GPU. Cái giá là chi phí attention tăng theo \(n^2\).`,
  example: 'Bản base đạt 27.3 BLEU trên WMT14 Anh–Đức sau 12 giờ train trên 8 GPU P100.',
  params: R`Bản base: \(d_{\text{model}} = 512\), \(h = 8\) đầu, \(d_{ff} = 2048\), \(N = 6\) lớp mỗi bên, khoảng 65M tham số. Các shape ở đây bỏ chiều batch \(B\) cho gọn.`,
  code: `model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6,
                       num_decoder_layers=6, dim_feedforward=2048,
                       dropout=0.1, batch_first=True)`
}, { label: 'Transformer', layout: 'manual', ref: [ref('vaswani', 'Bảng 3: cấu hình base')], children: [
  { ...tokSrc, pos: [-6.4, 2.6, 0] }, { ...embed('n', false), pos: [-4.55, 2.6, 0] }, { ...ENCODER, pos: [-1.95, 2.6, 0] },
  { ...tokTgt, pos: [-6.4, -2.6, 0] }, { ...embed('m', true), pos: [-4.55, -2.6, 0] }, { ...DECODER, pos: [-1.95, -2.6, 0] },
  { ...n('Linear → từ điển', 'linear', 'm×512 → m×V', {
    what: 'Chiếu vector 512 chiều ở mỗi vị trí thành V điểm số (logit), mỗi token trong từ điển một điểm.',
    formula: [R`\text{logits} = y' W_{\text{out}} + b, \qquad W_{\text{out}} = E^\top \in \mathbb{R}^{512 \times V}`],
    why: R`Khi dùng chung trọng số với embedding, logit của token \(k\) chính là tích vô hướng giữa trạng thái cuối và embedding của \(k\): mô hình chọn token có embedding "giống" trạng thái hiện tại nhất.`,
    params: R`\(512 \times 37\,000 \approx 18.9\)M nếu không dùng chung với embedding.`,
    code: `logits = F.linear(out, emb.weight)   # dùng chung E, (B, m, V)`
  }, { tag: 'Linear', size: [1.2, 1.6, 1.4] }), pos: [0.6, -2.6, 0] },
  { ...n('Softmax', 'act', 'm×V', {
    what: 'Biến logit thành phân bố xác suất trên toàn bộ từ điển cho token kế tiếp.',
    formula: [R`P(y_t = k \mid y_{\lt t}, x) = \frac{e^{z_k}}{\sum_{j=1}^{V} e^{z_j}}`, R`\mathcal{L} = -\frac{1}{m}\sum_{t=1}^{m} \log P(y_t \mid y_{\lt t}, x)`],
    why: 'Khi train, loss là cross-entropy với token đúng, kèm label smoothing 0.1 như bài gốc. Khi suy luận, token được chọn từ phân bố này: lấy lớn nhất (greedy), beam search (bài gốc dùng beam 4), hoặc lấy mẫu.',
    example: R`Logit \([2.0,\ 1.0,\ 0.1]\) → xác suất \([0.659,\ 0.242,\ 0.099]\).`,
    code: `loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), label_smoothing=0.1)
next_id = logits[:, -1].argmax(-1)   # greedy`
  }, { size: [0.3, 1.6, 1.4], ref: [ref('vaswani', 'mục 3.4 và 5.4')] }), pos: [2.3, -2.6, 0] }
], links: [
  { a: { i: 0, s: 'r' }, b: { i: 1, s: 'l' } }, { a: { i: 1, s: 'r' }, b: { i: 2, s: 'l' } },
  { a: { i: 3, s: 'r' }, b: { i: 4, s: 'l' } }, { a: { i: 4, s: 'r' }, b: { i: 5, s: 'l' } },
  { a: { i: 5, s: 'r' }, b: { i: 6, s: 'l' } }, { a: { i: 6, s: 'r' }, b: { i: 7, s: 'l' } },
  { a: { i: 7, s: 'r' }, b: { stub: 7, s: 'r' }, label: 'P(yₜ)' },
  { a: { i: 2, s: 'r' }, b: { i: 5, s: 't' }, kind: 'cross', label: 'H → K, V' }
] });
})();
