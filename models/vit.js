// ViT-B/16 (Dosovitskiy et al., 2020), torchvision vit_b_16. Pre-LN encoder on 196 image patches + [CLS].
(() => {
const C = { d: 768, h: 12, dk: 64, dff: 3072, act: 'gelu', drop: 0, noMask: true, noLive: true, L: 12 };
const T = '197'; // tokens: 196 patches + [CLS]
const pPatch = C.d * 3 * 16 * 16 + C.d, pPos = 197 * C.d;
const pLayer = 4 * C.d + (4 * C.d * C.d + 4 * C.d) + (2 * C.d * C.dff + C.dff + C.d);
const pHead = C.d * 1000 + 1000;
const pTotal = pPatch + C.d + pPos + C.L * pLayer + 2 * C.d + pHead;
// self-check: must match torchvision.models.vit_b_16 parameter count
console.assert(pTotal === 86567656, 'ViT-B/16 param count mismatch');

const encoder = n('Encoder', 'block', `${T}×${C.d} → ${T}×${C.d}`, {
  what: `${C.L} lớp encoder kiểu Pre-LN: LayerNorm đặt trước attention và trước MLP, residual cộng sau. MLP dùng GELU. Không có mặt nạ: mọi ô ảnh nhìn thấy mọi ô khác.`,
  formula: [R`z' = z + \mathrm{MHA}\big(\mathrm{LN}(z)\big)`, R`z_{\text{out}} = z' + \mathrm{MLP}\big(\mathrm{LN}(z')\big)`],
  why: 'Cùng khung với encoder của Transformer gốc, chỉ khác vị trí LayerNorm. Bài gốc đo khoảng cách chú ý: ngay ở các lớp đầu đã có đầu nhìn khắp ảnh lẫn đầu chỉ nhìn ô lân cận, trong khi CNN phải chồng nhiều lớp conv mới nhìn được xa như vậy.',
  params: R`Mỗi lớp \(${fmt(pLayer)}\): MHA \(${fmt(4 * C.d * C.d + 4 * C.d)}\), MLP \(${fmt(2 * C.d * C.dff + C.dff + C.d)}\), 2 LayerNorm \(${fmt(4 * C.d)}\). Cả ${C.L} lớp \(${fmt(C.L * pLayer)}\).`,
  code: `h = self.ln_1(x)
x = x + self.self_attention(h, h, h, need_weights=False)[0]
x = x + self.mlp(self.ln_2(x))`
}, { repeat: C.L, hint: `6 phần · ${C.L} lớp`, size: [2.0, 2.2, 1.6], ref: [ref('vit', 'mục 3.1 và hình 7')],
  children: [layerNorm(T, C.d), mha('self', T, T, C), addLeaf(T, C.d), layerNorm(T, C.d), ffn(T, C), addLeaf(T, C.d)],
  skips: [[0, 2, 'z'], [3, 5, "z′"]] });

MODELS.vit = n('ViT-B/16', 'block', '3×224×224 → 1000', {
  what: 'Transformer áp thẳng lên ảnh: cắt ảnh thành 196 ô 16×16, coi mỗi ô là một "từ", rồi cho chuỗi đó qua encoder của Transformer. Ngoài bước cắt ô, không có conv nào.',
  formula: [R`z_0 = [x_{\text{cls}};\ x_p^1 E;\ \dots;\ x_p^{196} E] + E_{\text{pos}}`, R`z_\ell = \mathrm{Encoder}_\ell(z_{\ell-1}), \qquad y = \mathrm{Head}\big(\mathrm{LN}(z_{12}^{0})\big)`],
  why: 'CNN cài sẵn hai giả định về ảnh: chỉ nhìn cục bộ, và một mẫu ở đâu trong ảnh cũng được xử lý như nhau. ViT gần như bỏ cả hai, để attention tự học quan hệ giữa mọi cặp ô ngay từ lớp đầu. Cái giá: thiếu giả định thì cần nhiều dữ liệu hơn. Chỉ train trên ImageNet-1k (1.3M ảnh), ViT thua ResNet cùng cỡ; pretrain trên JFT-300M (300M ảnh) thì ViT vượt.',
  example: R`\(224 / 16 = 14\) ô mỗi chiều → \(14 \times 14 = 196\) ô, thêm [CLS] là 197 token. Mỗi ô có \(16 \times 16 \times 3 = 768\) số, vừa bằng \(d = 768\) của ViT-Base. torchvision báo top-1 81.1% trên ImageNet (weights V1), ResNet-50 V2 là 80.9%.`,
  params: R`\(${fmt(pTotal)}\) (khoảng 86.6M): patch embedding \(${fmt(pPatch)}\), [CLS] và PE \(${fmt(C.d + pPos)}\), ${C.L} lớp × \(${fmt(pLayer)}\), LayerNorm cuối \(${fmt(2 * C.d)}\), head \(${fmt(pHead)}\). Cấu hình Base: \(d = ${C.d}\), ${C.h} đầu, MLP \(${C.dff}\), ${C.L} lớp.`,
  code: `from torchvision.models import vit_b_16, ViT_B_16_Weights
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval()
probs = model(x).softmax(-1)   # x: (B, 3, 224, 224) → (B, 1000)`
}, { label: 'ViT', noIn: true, outLabel: '1000 lớp', ref: [ref('vit')], children: [
  imageIn,
  n('Patch Embedding', 'embed', `3×224×224 → 196×${C.d}`, {
    what: `Cắt ảnh thành 196 ô 16×16 không chồng nhau và chiếu mỗi ô (768 số) thành một vector ${C.d} chiều.`,
    formula: [R`x_p^k \in \mathbb{R}^{16 \cdot 16 \cdot 3}, \qquad e_k = x_p^k E + b, \quad E \in \mathbb{R}^{768 \times ${C.d}}`],
    why: 'Một lớp Linear áp riêng cho từng ô cũng chính là một conv có kernel 16×16 và bước 16 (bằng đúng cỡ ô, nên các ô không chồng nhau). Cài đặt thật dùng conv vì nhanh hơn.',
    params: R`\(${C.d} \times 3 \times 16 \times 16 + ${C.d} = ${fmt(pPatch)}\).`,
    code: `x = self.conv_proj(x)              # (B, 768, 14, 14)
x = x.flatten(2).transpose(1, 2)   # (B, 196, 768)`
  }, { tag: 'Patch Embedding', size: [1.2, 1.5, 1.3], ref: [ref('vit', 'mục 3.1')], children: [
    n('Conv 16×16, bước 16', 'conv', `3×224×224 → ${C.d}×14×14`, {
      what: `Mỗi trong ${C.d} bộ lọc cỡ 3×16×16 đặt lên đúng một ô tại một thời điểm và nhảy 16 pixel, nên không ô nào bị nhìn hai lần.`,
      formula: CONV_F,
      why: 'Sau khi train, các bộ lọc này trông như những hàm cơ sở mô tả chi tiết bên trong một ô (hình 7 của bài gốc), khá giống lớp conv đầu tiên của CNN.',
      example: R`\(k = 16,\ p = 0,\ s = 16\): \(H_{out} = \lfloor (224 - 16) / 16 \rfloor + 1 = 14\).`,
      params: R`\(${fmt(pPatch)}\).`,
      code: `self.conv_proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)`
    }, { tag: 'Conv 16×16 /16', size: vol(C.d, 14) }),
    n('Duỗi thành chuỗi', 'shape', `${C.d}×14×14 → 196×${C.d}`, {
      what: 'Lưới 14×14 vị trí được duỗi thành chuỗi 196 token, đọc theo hàng từ trái sang phải, từ trên xuống dưới. Từ đây ảnh chỉ còn là một chuỗi, giống một câu 196 từ.',
      formula: [R`\mathbb{R}^{${C.d} \times 14 \times 14} \xrightarrow{\text{flatten}} \mathbb{R}^{${C.d} \times 196} \xrightarrow{\text{transpose}} \mathbb{R}^{196 \times ${C.d}}`],
      why: 'Thứ tự duỗi không quan trọng với attention (vốn không biết thứ tự). Vị trí 2D của từng ô chỉ còn được giữ nhờ PE ở bước sau.',
      code: `x = x.flatten(2).transpose(1, 2)`
    }, { tag: 'Duỗi', size: [0.6, 1.5, 1.3] })
  ] }),
  n('Thêm token [CLS]', 'embed', `196×${C.d} → ${T}×${C.d}`, {
    what: 'Ghép một vector học được vào đầu chuỗi. Token này không ứng với ô nào; qua 12 lớp nó gom thông tin từ mọi ô, và output của nó được dùng để phân loại.',
    formula: [R`z_0 = [x_{\text{cls}};\ e_1;\ \dots;\ e_{196}], \qquad x_{\text{cls}} \in \mathbb{R}^{${C.d}}`],
    why: 'Mượn từ BERT. Bài gốc thử cả cách lấy trung bình 196 output như Global Average Pooling của ResNet: kết quả tương đương nếu chỉnh learning rate phù hợp.',
    params: R`\(${C.d}\).`,
    code: `cls = self.class_token.expand(B, -1, -1)   # (B, 1, 768)
x = torch.cat([cls, x], dim=1)              # (B, 197, 768)`
  }, { tag: '[CLS]', size: [0.6, 1.5, 1.3], ref: [ref('vit', 'mục 3.1 và phụ lục D.3'), ref('bert')] }),
  n('Cộng vị trí (PE học được)', 'embed', `${T}×${C.d}`, {
    what: 'Cộng một vector vị trí cho mỗi trong 197 vị trí. Khác Transformer gốc, các vector này là tham số được học, không phải sin/cos.',
    formula: [R`z_0 \leftarrow z_0 + E_{\text{pos}}, \qquad E_{\text{pos}} \in \mathbb{R}^{${T} \times ${C.d}}`],
    why: 'PE một chiều học được tự khôi phục cấu trúc 2D: sau khi train, vector của các ô gần nhau trong ảnh giống nhau hơn, các ô cùng hàng hoặc cùng cột cũng giống nhau. Bài gốc thử PE 2D, kết quả không tốt hơn. Muốn fine-tune ở ảnh lớn hơn thì phải nội suy 2D bảng PE này.',
    params: R`\(${T} \times ${C.d} = ${fmt(pPos)}\).`,
    code: `x = x + self.pos_embedding   # nn.Parameter (1, 197, 768)`
  }, { tag: 'Vị trí (PE)', size: [0.9, 1.5, 1.3], ref: [ref('vit', 'phụ lục D.4')] }),
  encoder,
  { ...layerNorm(T, C.d), name: 'LayerNorm cuối' },
  n('Lấy token [CLS]', 'shape', `${T}×${C.d} → ${C.d}`, {
    what: 'Chỉ giữ output ở vị trí 0 (token [CLS]); 196 output của các ô bị bỏ đi.',
    formula: [R`y = z_{12}^{0}`],
    code: `x = x[:, 0]   # (B, 768)`
  }, { tag: 'Lấy [CLS]', size: [0.6, 1.5, 1.3] }),
  n('Head phân loại', 'linear', `${C.d} → 1000`, {
    what: 'Một lớp Linear cho ra 1000 logit, mỗi lớp của ImageNet một điểm.',
    formula: [R`\text{logits} = y W + b, \qquad W \in \mathbb{R}^{${C.d} \times 1000}`],
    why: 'Khi pretrain, bài gốc dùng MLP một lớp ẩn; khi fine-tune thì thay bằng một Linear duy nhất khởi tạo bằng 0. Muốn dùng ViT cho bài toán khác, thường chỉ cần thay lớp này.',
    params: R`\(${C.d} \times 1000 + 1000 = ${fmt(pHead)}\).`,
    code: `model.heads.head = nn.Linear(768, num_classes)   # thay đầu ra khi fine-tune`
  }, { tag: 'Head → 1000', size: [1.29, 0.22, 0.22] })
] });
})();
