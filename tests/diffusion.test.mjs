import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadPage } from './load.mjs';

const DIFF = loadPage()('DIFF');
const near = (got, want, tol, what) => want.forEach((w, i) => assert.ok(Math.abs(got[i] - w) < tol, `${what}[${i}]: ${got[i]} vs ${w}`));

test("the noise schedule is Stable Diffusion's: β from 0.00085 to 0.012, squared-linear, 1000 steps", () => {
  // ᾱ_t = Π (1 − β_s), computed separately in Python
  near([DIFF.AB[0], DIFF.AB[500], DIFF.AB[999]], [0.99915, 0.276333, 0.004660], 1e-6, 'ᾱ');
});

test('the sampler visits the same timesteps as the diffusers DDIM scheduler with steps_offset = 1', () => {
  const ts = DIFF.timesteps(50);
  assert.equal(ts.length, 50);
  assert.deepEqual([ts[0], ts[1], ts.at(-1)], [981, 961, 1]);
  assert.deepEqual([...DIFF.timesteps(5)], [801, 601, 401, 201, 1]);
});

test('the denoiser returns the exact posterior mean of a Gaussian mixture', () => {
  // one component μ = (1, 0), σ = 0.5, ᾱ = 0.25, x_t = (1.5, 0): v = ᾱσ² + 1 − ᾱ = 0.8125, k = √ᾱσ²/v = 0.153846
  // E[x₀ | x_t] = μ + k (x_t − √ᾱ μ) = 1 + 0.153846 · 1 = 1.153846; ε̂ = (x_t − √ᾱ x̂₀) / √(1 − ᾱ) = 1.065877
  near(DIFF.x0hat([1.5, 0], 0.25, [[1, 0]], 0.5), [1.153846, 0], 1e-6, 'x̂₀');
  near(DIFF.epsOf([1.5, 0], DIFF.x0hat([1.5, 0], 0.25, [[1, 0]], 0.5), 0.25), [1.065877, 0], 1e-6, 'ε̂');
  // two components at (±1, 0), x_t = (0, 0.3): equal responsibilities, the means cancel, only the pull k · 0.3 is left
  near(DIFF.x0hat([0, 0.3], 0.25, [[1, 0], [-1, 0]], 0.5), [0, 0.046154], 1e-6, 'symmetric x̂₀');
});

test('a DDIM step with the true noise lands exactly on the forward process at the earlier time', () => {
  // x₀ = (0.4, −0.2), ε = (1, 0.5): x_t at ᾱ = 0.3, one step to ᾱ′ = 0.6 must give √0.6 x₀ + √0.4 ε
  const x0 = [0.4, -0.2], e = [1, 0.5], xt = x0.map((v, d) => Math.sqrt(0.3) * v + Math.sqrt(0.7) * e[d]);
  near(DIFF.ddim(xt, e, 0.3, 0.6), [0.942294, 0.161308], 1e-6, 'x_t′');
});

test('guidance s = 0 is the empty prompt, s = 1 the prompt alone, and larger s extrapolates past it', () => {
  const x = [0.3, -0.4], ab = DIFF.AB[501];
  const u = DIFF.eps(x, ab, null, 0), c = DIFF.eps(x, ab, 'mèo', 1);
  near(DIFF.eps(x, ab, 'mèo', 0), u, 1e-12, 's = 0');
  near(DIFF.eps(x, ab, 'mèo', 3), u.map((v, d) => v + 3 * (c[d] - v)), 1e-12, 's = 3');
});

test('sampling with a prompt lands on that prompt\'s clusters; the empty prompt splits between both', () => {
  const share = (pts, prompt) => pts.filter(x => DIFF.nearest(x).prompt === prompt && DIFF.nearest(x).d < 3 * DIFF.S).length / pts.length;
  const cat = DIFF.sample({ prompt: 'mèo', s: 1 }), free = DIFF.sample({ prompt: null, s: 1 });
  assert.equal(cat.frames.length, 51);
  assert.ok(share(cat.frames.at(-1), 'mèo') > 0.9, `mèo, s = 1: ${share(cat.frames.at(-1), 'mèo')}`);
  const m = share(free.frames.at(-1), 'mèo'), d = share(free.frames.at(-1), 'chó');
  assert.ok(m > 0.3 && d > 0.3 && m + d > 0.9, `empty prompt: mèo ${m}, chó ${d}`);
});

test('stronger guidance pushes the samples away from the other prompt, and too much overshoots the clusters', () => {
  // "mèo" sits at x = −1, "chó" at x = +1. s = 1 samples the prompt's own distribution (mean x ≈ −1); larger s
  // extrapolates away from the empty prompt, so the points drift further left, past the clusters themselves
  const runs = Object.fromEntries([1, 3, 7.5, 15].map(s => [s, DIFF.sample({ prompt: 'mèo', s }).frames.at(-1)]));
  const meanX = s => runs[s].reduce((t, x) => t + x[0], 0) / runs[s].length;
  const meanD = s => runs[s].reduce((t, x) => t + DIFF.nearest(x).d, 0) / runs[s].length;
  assert.ok(Math.abs(meanX(1) + 1) < 0.05, `s = 1: mean x ${meanX(1)}`);
  assert.ok(meanX(1) > meanX(3) && meanX(3) > meanX(7.5) && meanX(7.5) > meanX(15), [1, 3, 7.5, 15].map(meanX).join(' > '));
  assert.ok(meanD(15) > 2 * meanD(1), `distance to the nearest cluster: s = 1 → ${meanD(1)}, s = 15 → ${meanD(15)}`);
});
