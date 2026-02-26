"""
prerez_mps.py — Apple MPS-accelerated variance-weighted tile SSIM

Drop-in replacement for the cv2-based SSIM functions in
prerez_extract.py. Falls back to CPU (numpy/cv2) if MPS
is unavailable.

Key optimisation: the 11×11 Gaussian blur at the heart of SSIM is a
2D convolution. On M1/M2/M3/M4 the GPU handles this ~5–10× faster than
the CPU, with zero transfer bottleneck thanks to unified memory.

The separable implementation (two 1×11 passes instead of one 11×11 pass)
further reduces ops by ~5×.

Additional gain: multiple tier roundtrips per frame are batched as
separate channels in a single GPU pass, amortising kernel launch overhead.

Usage:
    from prerez_mps import build_ssim_engine
    ssim_engine = build_ssim_engine(device="mps")   # or "cpu"

    # Drop-in for variance_weighted_ssim():
    score = ssim_engine.weighted(img_a, img_b,
                                  ref=img_a,
                                  tile_size=64,
                                  y_start=0, y_end=img_a.shape[0])

    # Batch multiple comparisons in one GPU pass:
    # pairs = [(img_a1, img_b1, ref1), (img_a2, img_b2, ref2), ...]
    scores = ssim_engine.weighted_batch(pairs, tile_size=64)
"""

from __future__ import annotations

import numpy as np

MIN_TILE_VARIANCE = 1.0


# ═══════════════════════════════════════════════════════════════════════
# CPU fallback (identical logic to v6.3 extractor)
# ═══════════════════════════════════════════════════════════════════════

def _gaussian_blur_cpu(img: np.ndarray, size: int = 11,
                        sigma: float = 1.5) -> np.ndarray:
    import cv2
    return cv2.GaussianBlur(img, (size, size), sigma)


def _ssim_map_cpu(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)
    mu1 = _gaussian_blur_cpu(i1)
    mu2 = _gaussian_blur_cpu(i2)
    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    s1  = _gaussian_blur_cpu(i1 * i1) - mu1_sq
    s2  = _gaussian_blur_cpu(i2 * i2) - mu2_sq
    s12 = _gaussian_blur_cpu(i1 * i2) - mu1_mu2
    return ((2*mu1_mu2+C1) * (2*s12+C2)) / ((mu1_sq+mu2_sq+C1) * (s1+s2+C2))


def _weighted_ssim_cpu(img_a: np.ndarray, img_b: np.ndarray,
                        ref: np.ndarray, tile_size: int,
                        y_start: int, y_end: int) -> float:
    ssim_map = _ssim_map_cpu(img_a, img_b)
    h, w = img_a.shape
    total_w = 0.0
    total_s = 0.0
    for y in range(y_start, y_end, tile_size):
        for x in range(0, w, tile_size):
            y2 = min(y + tile_size, y_end)
            x2 = min(x + tile_size, w)
            tile_ref  = ref[y:y2, x:x2].astype(np.float64)
            tile_ssim = ssim_map[y:y2, x:x2]
            v = tile_ref.var()
            if v < MIN_TILE_VARIANCE:
                continue
            total_s += v * float(tile_ssim.mean())
            total_w += v
    if total_w == 0:
        return float(ssim_map[y_start:y_end, :].mean())
    return total_s / total_w


# ═══════════════════════════════════════════════════════════════════════
# MPS engine
# ═══════════════════════════════════════════════════════════════════════

class SSIMEngine:
    """
    Wraps MPS or CPU SSIM computation behind a common interface.

    MPS path uses:
      - Separable 1×11 Gaussian conv (5× fewer ops than 11×11)
      - torch.nn.functional.unfold for vectorised tile variance
      - Batched multi-pair computation to amortise GPU launch cost
    """

    def __init__(self, device: str = "cpu"):
        self.device_str = device
        self._torch = None
        self._F = None
        self._kx = None
        self._ky = None

        if device == "mps":
            try:
                import torch
                import torch.nn.functional as F_mod
                if not torch.backends.mps.is_available():
                    raise RuntimeError("MPS not available")
                self._torch = torch
                self._F = F_mod
                self._device = torch.device("mps")
                self._kx, self._ky = self._make_kernels()
                self.device_str = "mps"
            except Exception as e:
                print(f"[prerez_mps] MPS unavailable ({e}), falling back to CPU")
                self.device_str = "cpu"
        else:
            self.device_str = "cpu"

    def _make_kernels(self):
        torch = self._torch
        size, sigma = 11, 1.5
        x = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-x**2 / (2 * sigma**2))
        g = g / g.sum()
        kx = g.view(1, 1, 1, size).to(self._device)
        ky = g.view(1, 1, size, 1).to(self._device)
        return kx, ky

    def _blur(self, t):
        """Separable Gaussian blur. t: (N,1,H,W) float32 on device."""
        F = self._F
        t = F.conv2d(t, self._kx, padding=(0, 5))
        t = F.conv2d(t, self._ky, padding=(5, 0))
        return t

    def _to_tensor(self, arr: np.ndarray):
        """numpy (H,W) float -> (1,1,H,W) float32 on MPS device."""
        torch = self._torch
        return (torch.from_numpy(arr.astype(np.float32))
                .unsqueeze(0).unsqueeze(0)
                .to(self._device))

    def _ssim_map_mps(self, img1: np.ndarray, img2: np.ndarray):
        """Returns (H,W) numpy SSIM map computed on MPS."""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        t1 = self._to_tensor(img1)
        t2 = self._to_tensor(img2)
        mu1 = self._blur(t1)
        mu2 = self._blur(t2)
        mu1_sq  = mu1 * mu1
        mu2_sq  = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        s1  = self._blur(t1 * t1) - mu1_sq
        s2  = self._blur(t2 * t2) - mu2_sq
        s12 = self._blur(t1 * t2) - mu1_mu2
        num = (2*mu1_mu2 + C1) * (2*s12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
        return (num / den).squeeze().cpu().numpy()

    def _tile_weighted_score(self, ssim_map: np.ndarray,
                              ref: np.ndarray, tile_size: int,
                              y_start: int, y_end: int) -> float:
        """
        Vectorised tile variance weighting using numpy stride tricks.
        ~3× faster than the Python tile loop for 1080p frames.
        """
        h, w = ref.shape
        tw = tile_size

        # Crop to mask region
        s_map = ssim_map[y_start:y_end, :]
        s_ref = ref[y_start:y_end, :]
        mh, mw = s_ref.shape

        # Pad to tile boundary
        pad_h = (tw - mh % tw) % tw
        pad_w = (tw - mw % tw) % tw
        if pad_h or pad_w:
            s_ref = np.pad(s_ref, ((0, pad_h), (0, pad_w)), mode='edge')
            s_map = np.pad(s_map, ((0, pad_h), (0, pad_w)),
                           constant_values=1.0)

        ph, pw = s_ref.shape
        nh = ph // tw
        nw = pw // tw

        # Reshape into tiles: (nh*nw, tw*tw)
        tiles_ref = (s_ref.reshape(nh, tw, nw, tw)
                          .transpose(0, 2, 1, 3)
                          .reshape(-1, tw * tw))
        tiles_map = (s_map.reshape(nh, tw, nw, tw)
                          .transpose(0, 2, 1, 3)
                          .reshape(-1, tw * tw))

        variances    = tiles_ref.var(axis=1)
        ssim_means   = tiles_map.mean(axis=1)
        valid        = variances >= MIN_TILE_VARIANCE

        if not valid.any():
            return float(s_map.mean())

        total_w = variances[valid].sum()
        total_s = (variances[valid] * ssim_means[valid]).sum()
        return float(total_s / total_w)

    # ── Public interface ───────────────────────────────────────────────

    def weighted(self, img_a: np.ndarray, img_b: np.ndarray,
                 ref: np.ndarray, tile_size: int = 64,
                 y_start: int = 0, y_end: int = None) -> float:
        """
        Drop-in for variance_weighted_ssim().
        img_a, img_b, ref: numpy (H,W) uint8 or float, same size.
        """
        if ref is None:
            ref = img_a
        if y_end is None:
            y_end = img_a.shape[0]

        if self.device_str == "mps":
            ssim_map = self._ssim_map_mps(img_a, img_b)
        else:
            ssim_map = _ssim_map_cpu(img_a, img_b)

        return self._tile_weighted_score(ssim_map, ref,
                                          tile_size, y_start, y_end)

    def weighted_batch(self, pairs: list[tuple],
                       tile_size: int = 64) -> list[float]:
        """
        Compute weighted SSIM for multiple (img_a, img_b, ref, y_start,
        y_end) tuples in one GPU pass (MPS) or sequentially (CPU).

        pairs: list of (img_a, img_b, ref, y_start, y_end)
        Returns: list of float scores in same order.
        """
        if not pairs:
            return []

        if self.device_str != "mps":
            return [
                _weighted_ssim_cpu(a, b, r, tile_size, y0, y1)
                for a, b, r, y0, y1 in pairs
            ]

        torch = self._torch
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # Stack all images into batches by resolution group
        # (pairs at the same resolution can be batched together)
        from collections import defaultdict
        by_shape = defaultdict(list)
        for idx, (a, b, r, y0, y1) in enumerate(pairs):
            by_shape[a.shape].append((idx, a, b, r, y0, y1))

        scores = [0.0] * len(pairs)

        for shape, group in by_shape.items():
            n = len(group)
            h, w = shape

            # Build (N, 1, H, W) tensors
            t1 = torch.stack([
                torch.from_numpy(g[1].astype(np.float32))
                for g in group]).unsqueeze(1).to(self._device)
            t2 = torch.stack([
                torch.from_numpy(g[2].astype(np.float32))
                for g in group]).unsqueeze(1).to(self._device)

            # Batch SSIM in one pass
            mu1 = self._blur(t1.view(1, n, h, w).view(n, 1, h, w))
            mu2 = self._blur(t2.view(n, 1, h, w))
            mu1_sq  = mu1*mu1; mu2_sq = mu2*mu2; mu1_mu2 = mu1*mu2
            s1  = self._blur(t1*t1) - mu1_sq
            s2  = self._blur(t2*t2) - mu2_sq
            s12 = self._blur(t1*t2) - mu1_mu2
            num = (2*mu1_mu2+C1)*(2*s12+C2)
            den = (mu1_sq+mu2_sq+C1)*(s1+s2+C2)
            ssim_maps = (num/den).squeeze(1).cpu().numpy()  # (N,H,W)

            for i, (idx, a, b, r, y0, y1) in enumerate(group):
                scores[idx] = self._tile_weighted_score(
                    ssim_maps[i], r, tile_size, y0, y1)

        return scores


# ═══════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════

def build_ssim_engine(device: str = "auto") -> SSIMEngine:
    """
    device: "auto" | "mps" | "cpu"
    "auto" uses MPS if available, otherwise CPU.
    """
    if device == "auto":
        try:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"[prerez_mps] Using device: {device}")
    return SSIMEngine(device=device)


# ═══════════════════════════════════════════════════════════════════════
# Benchmark (run directly to see speedup on your machine)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    print("Building engines...")
    cpu_engine = SSIMEngine(device="cpu")
    mps_engine = SSIMEngine(device="mps")

    N = 30
    results = {}

    for label, res in [("1080p", (1080, 1920)),
                        ("720p",  (720,  1280)),
                        ("480p",  (480,  854))]:
        h, w = res
        a   = np.random.randint(0, 255, (h, w), dtype=np.uint8).astype(np.float32)
        b   = np.random.randint(0, 255, (h, w), dtype=np.uint8).astype(np.float32)

        # Warmup MPS
        mps_engine.weighted(a, b, ref=a)

        t0 = time.perf_counter()
        for _ in range(N):
            cpu_engine.weighted(a, b, ref=a)
        cpu_ms = (time.perf_counter()-t0)/N*1000

        t0 = time.perf_counter()
        for _ in range(N):
            mps_engine.weighted(a, b, ref=a)
        import torch; torch.mps.synchronize()
        mps_ms = (time.perf_counter()-t0)/N*1000

        results[label] = (cpu_ms, mps_ms)
        print(f"{label:6s}  CPU: {cpu_ms:6.1f}ms   MPS: {mps_ms:6.1f}ms   "
              f"speedup: {cpu_ms/mps_ms:.1f}×")

    print()
    # Batch test: 5 pairs at 1080p (simulating one frame's tier comparisons)
    pairs_1080 = [
        (np.random.randint(0,255,(1080,1920),dtype=np.uint8).astype(np.float32),
         np.random.randint(0,255,(1080,1920),dtype=np.uint8).astype(np.float32),
         np.random.randint(0,255,(1080,1920),dtype=np.uint8).astype(np.float32),
         0, 1080)
        for _ in range(5)
    ]

    t0 = time.perf_counter()
    for _ in range(N):
        [cpu_engine.weighted(a,b,r,y_start=y0,y_end=y1)
         for a,b,r,y0,y1 in pairs_1080]
    cpu_batch_ms = (time.perf_counter()-t0)/N*1000

    t0 = time.perf_counter()
    for _ in range(N):
        mps_engine.weighted_batch(pairs_1080)
    torch.mps.synchronize()
    mps_batch_ms = (time.perf_counter()-t0)/N*1000

    print(f"Batch×5 1080p  CPU: {cpu_batch_ms:6.1f}ms   "
          f"MPS: {mps_batch_ms:6.1f}ms   "
          f"speedup: {cpu_batch_ms/mps_batch_ms:.1f}×")

    cpu_per_clip  = cpu_batch_ms  * 3   # 3 frames/clip, 5 SSIM ops/frame
    mps_per_clip  = mps_batch_ms  * 3
    print(f"\nEstimated per-clip SSIM time (3 frames, 5 ops each):")
    print(f"  CPU: {cpu_per_clip:.0f}ms")
    print(f"  MPS: {mps_per_clip:.0f}ms")
    print(f"  Saving {cpu_per_clip-mps_per_clip:.0f}ms/clip "
          f"× 1469 clips = "
          f"{(cpu_per_clip-mps_per_clip)*1469/1000/60:.1f} min")
