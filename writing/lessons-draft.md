> **DRAFT written by Claude from this project's history.** The findings and numbers are real and come from the
> commits and test runs; the wording is not yours. Rewrite it in your own words (and cut what you did not find
> interesting) before using it anywhere under your name.

# What building a neural-network explorer taught me about checking my own work

[visualize-dl](https://firstkhoi.github.io/visualize-dl/) is a 3D explorer of neural networks: Transformer, LLaMA 3,
ViT, Swin, ResNet, ConvNeXt, DenseNet, EfficientNet, LSTM, U-Net, Stable Diffusion, a small MLP that trains in the
page, and a CNN trained on MNIST that reads the digits you draw. Every block shows its formula, a worked example, its
parameter count and PyTorch code. Most numbers on the site are computed, not typed, and checked by tests.

The most useful thing I learned was not about any architecture. It was how often a check I trusted was wrong, and
how to find out.

## 1. A gradient check that failed when backprop was right

To train the MNIST CNN I wrote the backward pass by hand in NumPy and compared it with central differences before
training. The weights matched to 1e-8. The biases did not: some were off by 100%.

The bug was in the check. With the biases initialised to 0, every 3×3 window over MNIST's blank background has a
pre-activation of exactly 0, which is the kink of ReLU. Nudging a bias by +ε turns those units on and −ε turns them
off, so the loss has no derivative there and the finite difference lands between the two one-sided slopes. Checking
at random non-zero biases fixed it. Gradient checks assume a differentiable point; ReLU networks with zero biases on
sparse inputs sit on the one place where that assumption fails.

## 2. Tests that passed, or failed, for the wrong reason

- A browser test for "the chart tooltip survives a redraw" failed on the broken code, which looked like the expected
  red. It was failing for a different reason. The chart was redrawn eight times a second, so Playwright waited
  forever for it to stand still before hovering. The test would have failed on the fixed code too. I only noticed
  when the fix did not turn it green.
- A test for "weight decay leaves biases alone" passed even when I deliberately decayed the biases. They start at 0,
  and 0 × λ is still 0.

Since then every new test gets a *mutation check*: break the code on purpose and watch the test fail. It is the
only evidence that a test can catch what it is named after.

## 3. An overfitting detector that cried wolf

The training lab flips a share of the training labels and measures a clean test set, so it can show overfitting.
My first rule was "the test loss has climbed 20% above its lowest point while the train loss kept falling". With 20%
label noise and weight decay λ = 0.1 the network does *not* overfit (test accuracy stays near 92%), yet the rule
fired. The test loss had touched 0.178 at step 24, while the network was still a coarse blob, and then settled
around 0.25.

The signal that separates the two runs is the gap between test and train loss:

| | test loss | train loss | gap |
|---|---|---|---|
| no weight decay, step 2000 | 0.58 | 0.16 | 0.43 |
| λ = 0.1, step 2000 | 0.26 | 0.29 | −0.03 |

With noisy labels the train loss *cannot* go low unless the network memorises the noise. That is exactly what
overfitting is here, and the gap measures it directly.

## 4. Same code, different numbers

I ran the same 2000 training steps in Node and in Chromium, using the same JavaScript and the same seed. They agreed
exactly for 10 steps. At step 100 they differed in the last digit, at step 300 by 1e-14, at step 1000 by 1e-8, and by
step 2000 they were different runs: Chromium's lowest test loss came at step 1384, Node's at step 24.

Training with Adam is chaotic enough to amplify one rounding difference into a different trajectory. Both runs
still told the same story (both overfit), so the site quotes ranges instead of exact long-run numbers, and no test
asserts one. With weight decay the two engines stayed within 0.004 of each other, a nice illustration of
regularisation making training more stable.

## 5. Small things that were wrong for a while

- **The loss lagged one step behind the picture.** The lab reported the loss computed *before* each update, which
  is the usual PyTorch-loop habit. Meanwhile it drew the decision boundary from the weights *after* the update. In a
  loop that prints numbers this does not matter. Next to a picture it does.
- **L2 is not weight decay under Adam.** Adding λw to the gradient gets divided by Adam's running scale and mostly
  vanishes. Decoupled decay (AdamW) shrinks the weights directly.
- **Strong classifier-free guidance does not sharpen samples.** In the toy diffusion it pushes points *past* their
  clusters, a small version of the over-saturated images you get with a too-high guidance scale.
- **A layout audit beat looking.** A script opened all 702 blocks at five screen sizes with real formulas. It found
  the copy button covering 160 lines of code, and formulas inside sentences, which cannot wrap, pushing the whole
  panel sideways on phones in 80 cases. I had looked at the site many times without seeing either.

## 6. A real model, finally

Every demo used to run hand-picked toy numbers. The MNIST tab now runs a CNN I trained: two 3×3 convolutions,
max-pooling and a linear layer. It has 5 258 parameters and reaches 98.23% on the test set after about 2 minutes on
a CPU. The page reproduces the training logits to 1e-9.

The surprising part was the input, not the network. A digit drawn in a corner of the pad was often misread until the
drawing was prepared the way MNIST was: the ink's bounding box scaled into 20×20, then the centre of mass placed at
the centre of 28×28. The model only knows the world it was trained on.

## What I would tell myself at the start

Write the check before the code, and then check the check. Every bug above was caught by a test, an audit or a
measurement that disagreed with what I expected. The few that slipped through were the ones where I trusted a
number without asking how it could be wrong.
