// Navigation shared by the tab bar and the home page (index.html shows them): the models in groups, and learning paths
// through blocks of several models. A spec names a model, then children by name or tag: "unet/Conv 1×1 → 2 lớp".
const GROUPS = [
  ['CNN ảnh', ['resnet', 'convnext', 'densenet', 'efficientnet', 'cnn']],
  ['Transformer', ['transformer', 'llm', 'vit', 'swin']],
  ['Sinh ảnh', ['sd']],
  ['Chuỗi', ['lstm']],
  ['Y tế', ['unet', 'densenet/CheXNet']], // a spec with a path is a shortcut into another model's block
  ['Huấn luyện', ['mlp']]
];

// [label, spec]; a step without a spec is still to come
const PATHS = [
  { title: 'Phân vùng ảnh y khoa', goal: 'Từ một lát CT tới mặt nạ phân vùng, và cách chấm điểm mặt nạ đó cho đúng.', steps: [
    ['Cửa sổ HU cho ảnh CT', 'unet/Cửa sổ HU (tiền xử lý CT)'],
    ['U-Net: nhánh xuống và nhánh lên', 'unet'],
    ['Cắt và nối: đường tắt của U-Net', 'unet/Nhánh lên, tầng 1/Cắt và nối (copy & crop)'],
    ['Dice, cross-entropy và HD95', 'unet/Conv 1×1 → 2 lớp'],
    ['nnU-Net'], ['Swin UNETR'], ['MedSAM']
  ] },
  { title: 'Chẩn đoán X-quang ngực', goal: 'Dùng lại một CNN train trên ImageNet để dò 14 bệnh trên phim ngực.', steps: [
    ['DenseNet-121', 'densenet'],
    ['Dense block: nối thay vì cộng', 'densenet/Dense block 1'],
    ['CheXNet: 14 nhãn thay cho 1000 lớp', 'densenet/CheXNet'],
    ['Sigmoid cho ảnh nhiều bệnh', 'densenet/CheXNet/Sigmoid từng nhãn'],
    ['BCE có trọng số cho bệnh hiếm', 'densenet/CheXNet/BCE có trọng số'],
    ['Grad-CAM'], ['CNN trên MedMNIST']
  ] },
  { title: 'Attention từ con số', goal: 'Self-attention với số chạy thật, rồi những gì các LLM hiện nay đổi đi.', steps: [
    ['Self-attention, số chạy thật', 'transformer/Encoder/Multi-Head Self-Attention/Scaled Dot-Product Attention'],
    ['Mặt nạ nhân quả', 'transformer/Decoder/Masked Multi-Head Attention/Scaled Dot-Product Attention/Mặt nạ'],
    ['Đường tắt residual và gradient', 'transformer/Encoder/Add & Norm/Cộng residual'],
    ['LLaMA 3: RoPE, GQA và KV cache', 'llm/Decoder/Grouped-Query Attention'],
    ['Nhiệt độ và top-p khi sinh chữ', 'llm/Lấy mẫu token kế'],
    ['ViT: ảnh thành chuỗi ô', 'vit/Patch Embedding'],
    ['Swin: attention trong cửa sổ dịch', 'swin/Tầng 1/Khối Swin (SW-MSA)']
  ] },
  { title: 'CNN từ con số', goal: 'Từ một bộ lọc 3×3 trên chữ số viết tay tới các mạng ImageNet.', steps: [
    ['Vẽ một chữ số cho CNN', 'cnn'],
    ['Tích chập 3×3 và feature map', 'cnn/Conv 3×3, 8 kênh'],
    ['ResNet: bottleneck và đường tắt', 'resnet/Tầng conv2_x/Bottleneck đầu (projection)'],
    ['EfficientNet: conv depthwise', 'efficientnet'],
    ['ConvNeXt: CNN học từ Transformer', 'convnext']
  ] },
  { title: 'Một mạng học như thế nào', goal: 'Train một mạng nhỏ ngay trong trang, xem nó overfit, và xem gradient đi qua mạng sâu.', steps: [
    ['Phòng thí nghiệm huấn luyện', 'mlp'],
    ['Backprop qua một lớp Linear', 'mlp/Linear 2 → H'],
    ['Softmax và cross-entropy', 'mlp/Softmax + Cross-entropy'],
    ['Gradient qua 100 lớp, có và không có đường tắt', 'resnet/Tầng conv2_x/Bottleneck (identity)/Cộng đường tắt'],
    ['Gradient qua thời gian: RNN và LSTM', 'lstm/LSTM']
  ] }
];

// the link (#model.i.j) of a spec, or null when it names nothing
function navCode(spec) {
  const [model, ...names] = spec.split('/'), code = [model];
  let nd = MODELS[model];
  for (const name of names) {
    const i = nd?.children?.findIndex(k => k.name === name || k.tag === name) ?? -1;
    if (i < 0) return null;
    code.push(i);
    nd = nd.children[i];
  }
  return nd ? code.join('.') : null;
}
