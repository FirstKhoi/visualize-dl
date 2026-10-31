# Training the CNN of the "CNN (MNIST)" tab

The page runs a small CNN that was trained for real. This folder is how it was trained, so anyone can check or redo it.

```
conv 3×3, 8 channels → ReLU → max-pool 2 → conv 3×3, 16 channels → ReLU → max-pool 2 → linear 400 → 10
```

| Layer | Output | Parameters |
|---|---|---|
| input | 1 × 28 × 28 | |
| conv1 3×3 + ReLU | 8 × 26 × 26 | 8·1·9 + 8 = 80 |
| max-pool 2×2 | 8 × 13 × 13 | |
| conv2 3×3 + ReLU | 16 × 11 × 11 | 16·8·9 + 16 = 1 168 |
| max-pool 2×2 | 16 × 5 × 5 (the 11th row and column are dropped) | |
| linear | 10 | 10·400 + 10 = 4 010 |
| **total** | | **5 258** |

## Run it

```sh
python3 training/train_mnist_cnn.py   # NumPy only; downloads MNIST into training/data/ (gitignored)
npm test                              # the page's forward pass must reproduce the new logits
./build-pages.sh                      # copy the new weights into docs/
```

It writes `models/weights/mnist-cnn.js` (the weights the page loads), `tests/fixtures/mnist-cnn.json` (five test images and their logits, which the page's JavaScript forward pass must reproduce to 1e-9) and `results.json`.

`mnist_cnn_pytorch.py` is the same model in PyTorch, for Google Colab (`pip install torch torchvision`). It writes the same files in the same format. The environment that produced the weights in this repository had no PyTorch, so that script has not been run yet: check its numbers the first time.

## Result

Seed 0, Adam (η = 0.001), batch 64, every training image shifted by up to 2 pixels, about 2 minutes on a laptop CPU:

| Epoch | Train loss | Train acc. | Test loss | Test acc. |
|---|---|---|---|---|
| 1 | 0.451 | 86.1% | 0.116 | 96.58% |
| 2 | 0.189 | 94.1% | 0.082 | 97.44% |
| 3 | 0.146 | 95.5% | 0.072 | 97.62% |
| 4 | 0.131 | 96.0% | 0.066 | 98.05% |
| 5 | 0.115 | 96.5% | 0.067 | 97.94% |
| 6 | 0.106 | 96.8% | 0.057 | 98.23% |

The page gets 5 significant digits per weight; with exactly those weights the test accuracy is still **98.23%**. Training accuracy is lower than test accuracy because it is measured on the shifted images, which are harder.

## Choices, and why

- **NumPy with hand-written backprop instead of PyTorch.** Every gradient is visible in about 60 lines, the same way the page's training lab computes its gradients, and the script runs anywhere. Before training, backprop is compared with central differences on every parameter.
- **The gradient check runs with random, non-zero biases.** With the biases at 0, every 3×3 window over the blank background gives a pre-activation of exactly 0, which is the kink of ReLU. Nudging a bias by ±ε switches those units on or off, so the loss is not differentiable there, and the first version of the check failed on the biases even though backprop was right.
- **No normalisation of the input, pixels in [0, 1].** The page feeds the drawing as-is, so the network sees what you drew without any extra step.
- **Random shifts of up to 2 px.** Hand-drawn digits are never centred as exactly as MNIST's.
- **The drawing is prepared the way MNIST was** (`CNN.prepare` in `live.js`):
  1. scale the ink's bounding box into 20×20, keeping the aspect ratio;
  2. place it in 28×28 with its centre of mass in the middle.

  Without this a digit drawn in a corner is often misread.
- **Weights in PyTorch's `state_dict` layout** (conv: out × in × 3 × 3, linear: out × in, flatten channel first), so a model trained in PyTorch exports the same way and the page needs no change.

MNIST: LeCun, Cortes & Burges, <http://yann.lecun.com/exdb/mnist/>, CC BY-SA 3.0. The page ships ten of its test images as samples.
