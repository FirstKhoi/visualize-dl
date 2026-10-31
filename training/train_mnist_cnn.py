"""Trains the small CNN of the page's "CNN (MNIST)" tab with NumPy only.

    conv 3×3, 8 channels → ReLU → max-pool 2 → conv 3×3, 16 channels → ReLU → max-pool 2 → linear 400 → 10

Forward and backward passes are written by hand (no autograd) and checked against finite differences before
training. The script writes:

  models/weights/mnist-cnn.js   the weights, in PyTorch's state_dict layout (conv: out × in × 3 × 3,
                                linear: out × in), so a model trained in PyTorch exports the same way
  tests/fixtures/mnist-cnn.json test images with the logits the page's JS forward pass must reproduce
  training/results.json         loss and accuracy per epoch

Run from the repository root:  python3 training/train_mnist_cnn.py
MNIST: LeCun, Cortes & Burges, http://yann.lecun.com/exdb/mnist/ (CC BY-SA 3.0), via torchvision's mirror.
"""
import base64
import gzip
import json
import pathlib
import struct
import time
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / 'training' / 'data'
MIRROR = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
FILES = {'train_x': 'train-images-idx3-ubyte.gz', 'train_y': 'train-labels-idx1-ubyte.gz',
         'test_x': 't10k-images-idx3-ubyte.gz', 'test_y': 't10k-labels-idx1-ubyte.gz'}
SEED, EPOCHS, BATCH, LR = 0, 6, 64, 1e-3


def load(name):
    """MNIST in its original IDX format: images as uint8 (n, 28, 28), labels as uint8 (n,)."""
    path = DATA / FILES[name]
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MIRROR + FILES[name], path)
    raw = gzip.decompress(path.read_bytes())
    if name.endswith('_x'):
        n, h, w = struct.unpack('>III', raw[4:16])
        return np.frombuffer(raw, np.uint8, offset=16).reshape(n, h, w)
    return np.frombuffer(raw, np.uint8, offset=8)


# ---------- layers: forward returns what backward needs ----------

def conv_forward(x, w, b):
    """3×3 cross-correlation, stride 1, no padding (nn.Conv2d): (B, C, H, W) → (B, O, H − 2, W − 2)."""
    win = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(2, 3))  # B, C, H', W', 3, 3
    B, C, Ho, Wo = win.shape[:4]
    cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(B * Ho * Wo, C * 9)  # one row per output position
    out = cols @ w.reshape(len(w), -1).T + b
    return out.reshape(B, Ho, Wo, -1).transpose(0, 3, 1, 2), cols


def conv_backward(dout, x_shape, cols, w):
    B, O, Ho, Wo = dout.shape
    d = dout.transpose(0, 2, 3, 1).reshape(-1, O)
    dw, db = (d.T @ cols).reshape(w.shape), d.sum(0)
    dwin = (d @ w.reshape(O, -1)).reshape(B, Ho, Wo, x_shape[1], 3, 3)
    dx = np.zeros(x_shape)
    for i in range(3):  # each input pixel collects the gradient of every window it sat in
        for j in range(3):
            dx[:, :, i:i + Ho, j:j + Wo] += dwin[..., i, j].transpose(0, 3, 1, 2)
    return dx, dw, db


def pool_forward(x):
    """2×2 max-pool, stride 2; an odd last row or column is dropped (nn.MaxPool2d's floor)."""
    B, C, H, W = x.shape
    h, w = H // 2, W // 2
    win = x[:, :, :2 * h, :2 * w].reshape(B, C, h, 2, w, 2).transpose(0, 1, 2, 4, 3, 5).reshape(B, C, h, w, 4)
    arg = win.argmax(-1)  # the first maximum, as PyTorch picks
    return np.take_along_axis(win, arg[..., None], -1)[..., 0], arg


def pool_backward(dout, arg, x_shape):
    B, C, h, w = dout.shape
    dwin = np.zeros((B, C, h, w, 4))
    np.put_along_axis(dwin, arg[..., None], dout[..., None], -1)  # only the maximum passes the gradient back
    dx = np.zeros(x_shape)
    dx[:, :, :2 * h, :2 * w] = dwin.reshape(B, C, h, w, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(B, C, 2 * h, 2 * w)
    return dx


# ---------- the network ----------

def init(rng):
    """He initialisation, biases 0. Keys and shapes are PyTorch's state_dict for the same nn.Module."""
    he = lambda *shape, fan_in: rng.normal(0, np.sqrt(2 / fan_in), shape)
    return {'conv1.weight': he(8, 1, 3, 3, fan_in=9), 'conv1.bias': np.zeros(8),
            'conv2.weight': he(16, 8, 3, 3, fan_in=72), 'conv2.bias': np.zeros(16),
            'fc.weight': he(10, 400, fan_in=400), 'fc.bias': np.zeros(10)}


def forward(p, x):
    """x: (B, 1, 28, 28) in [0, 1] → logits (B, 10), plus what the backward pass needs."""
    z1, c1 = conv_forward(x, p['conv1.weight'], p['conv1.bias'])
    m1, i1 = pool_forward(np.maximum(z1, 0))
    z2, c2 = conv_forward(m1, p['conv2.weight'], p['conv2.bias'])
    m2, i2 = pool_forward(np.maximum(z2, 0))
    f = m2.reshape(len(x), -1)  # 16 × 5 × 5 = 400, channel first, as torch.flatten
    return f @ p['fc.weight'].T + p['fc.bias'], (x, z1, c1, i1, m1, z2, c2, i2, m2, f)


def loss_grad(p, x, y):
    """Mean softmax cross-entropy, accuracy and the gradient of every parameter."""
    logits, (x, z1, c1, i1, m1, z2, c2, i2, m2, f) = forward(p, x)
    e = np.exp(logits - logits.max(1, keepdims=True))
    prob = e / e.sum(1, keepdims=True)
    n = len(y)
    loss = -np.log(prob[np.arange(n), y] + 1e-12).mean()
    d = prob.copy()
    d[np.arange(n), y] -= 1  # softmax + cross-entropy: p − onehot(y)
    d /= n
    g = {'fc.weight': d.T @ f, 'fc.bias': d.sum(0)}
    dm2 = (d @ p['fc.weight']).reshape(m2.shape)
    dz2 = pool_backward(dm2, i2, z2.shape) * (z2 > 0)
    dm1, g['conv2.weight'], g['conv2.bias'] = conv_backward(dz2, m1.shape, c2, p['conv2.weight'])
    dz1 = pool_backward(dm1, i1, z1.shape) * (z1 > 0)
    _, g['conv1.weight'], g['conv1.bias'] = conv_backward(dz1, x.shape, c1, p['conv1.weight'])
    return loss, (logits.argmax(1) == y).mean(), g


def gradient_check(rng, x, y):
    """Backprop must match central differences on a few entries of every parameter."""
    p = init(rng)
    # with zero biases every window over the blank background gives a pre-activation of exactly 0, the kink of
    # ReLU, where nudging a bias by ±ε switches those units on or off: the loss is not differentiable there
    for k in p:
        if k.endswith('bias'):
            p[k] = rng.normal(0, 0.1, p[k].shape)
    _, _, g = loss_grad(p, x, y)
    worst = 0
    for k, v in p.items():
        for idx in [tuple(rng.integers(0, s) for s in v.shape) for _ in range(4)]:
            old, eps = v[idx], 1e-6
            v[idx] = old + eps; lp = loss_grad(p, x, y)[0]
            v[idx] = old - eps; lm = loss_grad(p, x, y)[0]
            v[idx] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - g[k][idx]) / max(1e-8, abs(num) + abs(g[k][idx])))
    assert worst < 1e-5, f'backprop does not match finite differences ({worst:.2e})'
    return worst


def shift(x, rng, most=2):
    """Moves each image by up to `most` pixels in x and y (zeros come in): hand-drawn digits are rarely centred exactly."""
    out = np.zeros_like(x)
    for k, (dy, dx) in enumerate(rng.integers(-most, most + 1, (len(x), 2))):
        src = x[k, 0, max(0, -dy):28 - max(0, dy), max(0, -dx):28 - max(0, dx)]
        out[k, 0, max(0, dy):max(0, dy) + src.shape[0], max(0, dx):max(0, dx) + src.shape[1]] = src
    return out


def evaluate(p, x, y, size=1000):
    hits, loss = 0, 0.0
    for s in range(0, len(x), size):
        logits, _ = forward(p, x[s:s + size])
        e = np.exp(logits - logits.max(1, keepdims=True))
        prob = e / e.sum(1, keepdims=True)
        loss -= np.log(prob[np.arange(len(logits)), y[s:s + size]] + 1e-12).sum()
        hits += (logits.argmax(1) == y[s:s + size]).sum()
    return loss / len(x), hits / len(x)


def main():
    rng = np.random.default_rng(SEED)
    tx, ty, vx, vy = load('train_x'), load('train_y').astype(int), load('test_x'), load('test_y').astype(int)
    to_input = lambda imgs: (imgs / 255.0)[:, None]  # pixels to [0, 1]; the page feeds its drawing the same way
    X, V = to_input(tx), to_input(vx)
    print(f'gradient check: worst relative error {gradient_check(np.random.default_rng(1), X[:4], ty[:4]):.1e}')

    p = init(rng)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    s = {k: np.zeros_like(v) for k, v in p.items()}
    t, history, start = 0, [], time.time()
    for epoch in range(1, EPOCHS + 1):
        order, losses, accs = rng.permutation(len(X)), [], []
        for b in range(0, len(X), BATCH):
            idx = order[b:b + BATCH]
            loss, acc, g = loss_grad(p, shift(X[idx], rng), ty[idx])
            t += 1
            for k in p:  # Adam
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                s[k] = 0.999 * s[k] + 0.001 * g[k] ** 2
                p[k] -= LR * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(s[k] / (1 - 0.999 ** t)) + 1e-8)
            losses.append(loss); accs.append(acc)
        test_loss, test_acc = evaluate(p, V, vy)
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'train_acc': float(np.mean(accs)),
                        'test_loss': float(test_loss), 'test_acc': float(test_acc), 'seconds': round(time.time() - start, 1)})
        print(history[-1])

    # the page gets 5 significant digits per weight; everything below is measured with exactly those weights
    rounded = {k: np.array([float(f'{w:.5g}') for w in v.ravel()]).reshape(v.shape) for k, v in p.items()}
    test_loss, test_acc = evaluate(rounded, V, vy)
    logits, _ = forward(rounded, V)
    first = [int(np.flatnonzero((vy == d) & (logits.argmax(1) == d))[0]) for d in range(10)]  # one test image per digit
    n_params = sum(v.size for v in p.values())
    print(f'{n_params} parameters; test accuracy with the exported weights: {test_acc:.4f}')

    weights = {k: {'shape': list(v.shape), 'data': v.ravel().tolist()} for k, v in rounded.items()}
    meta = {'params': int(n_params), 'test_acc': float(test_acc), 'test_loss': float(test_loss), 'epochs': EPOCHS,
            'samples': [{'label': int(vy[i]), 'pixels': base64.b64encode(vx[i].tobytes()).decode()} for i in first]}
    out = ROOT / 'models' / 'weights' / 'mnist-cnn.js'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('// Generated by training/train_mnist_cnn.py: do not edit by hand.\n'
                   '// A small CNN trained on MNIST (LeCun, Cortes & Burges, CC BY-SA 3.0); weights in PyTorch state_dict layout,\n'
                   '// samples: one MNIST test image per digit (28×28 uint8, base64).\n'
                   f'const MNIST_CNN = {json.dumps({"weights": weights, **meta}, separators=(",", ":"))};\n')
    fixtures = {'images': [vx[i].ravel().tolist() for i in range(5)], 'labels': vy[:5].tolist(),
                'logits': logits[:5].tolist(), 'test_acc': float(test_acc)}
    (ROOT / 'tests' / 'fixtures').mkdir(exist_ok=True)
    (ROOT / 'tests' / 'fixtures' / 'mnist-cnn.json').write_text(json.dumps(fixtures) + '\n')
    (ROOT / 'training' / 'results.json').write_text(json.dumps({'seed': SEED, 'epochs': EPOCHS, 'batch': BATCH, 'lr': LR,
        'augmentation': 'random shift up to 2 px', 'params': int(n_params), 'history': history,
        'exported_test_acc': float(test_acc), 'exported_test_loss': float(test_loss)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
