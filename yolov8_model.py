import torch
import torch.nn as nn
import math


# ==================== Utils ====================
def make_divisible(x, divisor=8):
    """Ensure channel count is divisible by divisor."""
    return int((x + divisor / 2) // divisor * divisor)


class YOLOScale:
    """YOLOv8 depth and width scaling."""
    def __init__(self, depth=0.67, width=0.75):
        self.d = depth
        self.w = width

    def c(self, ch):
        """Scale channel count."""
        return make_divisible(ch * self.w)

    def n(self, n):
        """Scale depth (number of layers)."""
        return max(1, int(round(n * self.d)))


# ==================== Basic Blocks ====================
class Conv(nn.Module):
    """Standard Conv2d + BatchNorm + SiLU activation."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s,
            padding=k // 2 if p is None else p,
            groups=g,
            bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """YOLOv8 bottleneck with residual connection."""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """YOLOv8 C2f block with split-concatenate architecture."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


# ==================== Detection Head ====================
class Detect(nn.Module):
    """YOLOv8 detection head with decoupled classification and regression."""
    def __init__(self, nc=80, ch=(256, 512, 1024), reg_max=16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.no = nc + reg_max * 4  # outputs per anchor
        self.stride = torch.tensor([8., 16., 32.])

        c2 = max(ch[0] // 4, reg_max * 4, 16)
        c3 = max(ch[0], min(nc * 2, 128))

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * reg_max, 1)
            ) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),
                Conv(c3, c3, 3),
                nn.Conv2d(c3, nc, 1)
            ) for x in ch
        )

        self._init_bias()

    def _init_bias(self):
        """Initialize detection head biases."""
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)

    def forward(self, x):
        """Forward pass returning [bbox_reg, cls] concatenated per scale."""
        return [torch.cat([self.cv2[i](x[i]), self.cv3[i](x[i])], 1) for i in range(len(x))]


# ==================== YOLOv8 Model ====================
class YOLOv8(nn.Module):
    """
    YOLOv8 Medium (YOLOv8m) Implementation
    Output: P3: 80×80×256, P4: 40×40×512, P5: 20×20×1024
    """
    def __init__(self, nc=80, depth=0.67, width=0.75):
        super().__init__()
        self.nc = nc
        s = YOLOScale(depth, width)

        # Channel dimensions - FIXED for correct output channels
        c1 = s.c(64)   # 48
        c2 = s.c(128)  # 96
        c3 = s.c(256)  # 192
        c4 = s.c(512)  # 384
        c5 = s.c(1024)  # 768 

        # Output channels (what we want)
        out_c3 = 256  # P3 output
        out_c4 = 512  # P4 output
        out_c5 = 1024 # P5 output

        # ========== Backbone ==========
        self.b0 = Conv(3, c1, 3, 2)              # P1: 640×640×3 → 320×320×48
        self.b1 = C2f(c1, c1, s.n(3), True)      #     320×320×48 → 320×320×48

        self.b2 = Conv(c1, c2, 3, 2)             # P2: 320×320×48 → 160×160×96
        self.b3 = C2f(c2, c2, s.n(6), True)      #     160×160×96 → 160×160×96

        self.b4 = Conv(c2, c3, 3, 2)             # P3: 160×160×96 → 80×80×192
        self.b5 = C2f(c3, c3, s.n(6), True)      #     80×80×192 → 80×80×192

        self.b6 = Conv(c3, c4, 3, 2)             # P4: 80×80×192 → 40×40×384
        self.b7 = C2f(c4, c4, s.n(3), True)      #     40×40×384 → 40×40×384

        self.b8 = Conv(c4, c5, 3, 2)             # P5: 40×40×384 → 20×20×384
        self.b9 = C2f(c5, c5, s.n(3), True)      #     20×20×384 → 20×20×384
        self.b10 = SPPF(c5, c5, 5)               #     20×20×384 → 20×20×384

        # ========== Neck (PAN-FPN) ==========
        # Top-down pathway
        self.n0 = nn.Upsample(None, 2, 'nearest')  # 20×20 → 40×40
        self.n1 = C2f(c4 + c5, c4, s.n(3), False)  # 40×40×(384+384) → 40×40×384

        self.n2 = nn.Upsample(None, 2, 'nearest')  # 40×40 → 80×80
        self.n3 = C2f(c3 + c4, out_c3, s.n(3), False)  # 80×80×(192+384) → 80×80×256 (P3)

        # Bottom-up pathway
        self.n4 = Conv(out_c3, out_c3, 3, 2)           # 80×80×256 → 40×40×256
        self.n5 = C2f(out_c3 + c4, out_c4, s.n(3), False)  # 40×40×(256+384) → 40×40×512 (P4)

        self.n6 = Conv(out_c4, out_c4, 3, 2)           # 40×40×512 → 20×20×512
        self.n7 = C2f(out_c4 + c5, out_c5, s.n(3), False)  # 20×20×(512+384) → 20×20×1024 (P5)

        # ========== Detection Head ==========
        self.head = Detect(nc, (out_c3, out_c4, out_c5), 16)

    def forward(self, x):
        """
        Forward pass.
        Input: [B, 3, 640, 640]
        Output: List of 3 tensors (P3, P4, P5)
            P3: [B, nc+64, 80, 80]   (stride 8)  - 256 channels
            P4: [B, nc+64, 40, 40]   (stride 16) - 512 channels
            P5: [B, nc+64, 20, 20]   (stride 32) - 1024 channels
        """
        # Backbone
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        p3 = self.b5(x)  # 80×80×192
        x = self.b6(p3)
        p4 = self.b7(x)  # 40×40×384
        x = self.b8(p4)
        x = self.b9(x)
        p5 = self.b10(x)  # 20×20×384

        # Neck - Top-down
        x = self.n0(p5)                          # Upsample
        x = torch.cat([x, p4], 1)                # Concat
        x = self.n1(x)                           # C2f
        p4_out = x                               # Save for PAN

        x = self.n2(x)                           # Upsample
        x = torch.cat([x, p3], 1)                # Concat
        p3_out = self.n3(x)                      # C2f → P3 output (256 channels)

        # Neck - Bottom-up
        x = self.n4(p3_out)                      # Downsample
        x = torch.cat([x, p4_out], 1)            # Concat
        p4_out = self.n5(x)                      # C2f → P4 output (512 channels)

        x = self.n6(p4_out)                      # Downsample
        x = torch.cat([x, p5], 1)                # Concat
        p5_out = self.n7(x)                      # C2f → P5 output (1024 channels)

        # Detection head
        return self.head([p3_out, p4_out, p5_out])


# ==================== Test Code ====================
if __name__ == "__main__":
    # YOLOv8m configuration
    model = YOLOv8(nc=80, depth=0.67, width=0.75)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    outputs = model(x)
    
    print("\nOutput shapes:")
    for i, out in enumerate(outputs):
        stride = 8 * (2 ** i)
        print(f"  P{i+3} (stride {stride:2d}): {list(out.shape)}")
    
    print("\nFeature map breakdown:")
    print("  Scale | Stride | Resolution | Channels | Output Shape")
    print("  ------|--------|------------|----------|---------------")
    for i, (out, stride) in enumerate(zip(outputs, [8, 16, 32])):
        b, c, h, w = out.shape
        print(f"  P{i+3}   |   {stride:2d}   | {h:3d}×{w:3d}   |   {c:3d}    | {list(out.shape)}")
    
    # Verify channel dimensions
    print("\n✅ Channel Verification:")
    print(f"  P3 feature channels: {outputs[0].shape[1]} (expected: 80+64=144)")
    print(f"  P4 feature channels: {outputs[1].shape[1]} (expected: 80+64=144)")
    print(f"  P5 feature channels: {outputs[2].shape[1]} (expected: 80+64=144)")
