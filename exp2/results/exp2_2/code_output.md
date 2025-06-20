```
loaded image path: love.jpg
input tensor shape: torch.Size([1, 1, 28, 28])
input image size: 28 × 28
kernel size: 3 × 3

combination 1: no padding, stride 1
parameters: padding=0, stride=1
theoretical calculation:
  output height = (28 + 2×0 - 3) ÷ 1 + 1 = 26
  output width = (28 + 2×0 - 3) ÷ 1 + 1 = 26
  theoretical output shape: 26 × 26
actual validation:
  actual output tensor shape: torch.Size([1, 1, 26, 26])
  actual output shape: 26 × 26
validation result:
  height match: True (theoretical: 26, actual: 26)
  width match: True (theoretical: 26, actual: 26)
  overall match: True
--------------------------------------------------
combination 2: padding 1, stride 2
parameters: padding=1, stride=2
theoretical calculation:
  output height = (28 + 2×1 - 3) ÷ 2 + 1 = 14
  output width = (28 + 2×1 - 3) ÷ 2 + 1 = 14
  theoretical output shape: 14 × 14
actual validation:
  actual output tensor shape: torch.Size([1, 1, 14, 14])
  actual output shape: 14 × 14
validation result:
  height match: True (theoretical: 14, actual: 14)
  width match: True (theoretical: 14, actual: 14)
  overall match: True
--------------------------------------------------
combination 3: padding 2, stride 3
parameters: padding=2, stride=3
theoretical calculation:
  output height = (28 + 2×2 - 3) ÷ 3 + 1 = 10
  output width = (28 + 2×2 - 3) ÷ 3 + 1 = 10
  theoretical output shape: 10 × 10
actual validation:
  actual output tensor shape: torch.Size([1, 1, 10, 10])
  actual output shape: 10 × 10
validation result:
  height match: True (theoretical: 10, actual: 10)
  width match: True (theoretical: 10, actual: 10)
  overall match: True
--------------------------------------------------

============================================================
experiment results summary
============================================================
combination 1: no padding, stride 1: ✓ match
  theoretical: (26, 26), actual: (26, 26)
combination 2: padding 1, stride 2: ✓ match
  theoretical: (14, 14), actual: (14, 14)
combination 3: padding 2, stride 3: ✓ match
  theoretical: (10, 10), actual: (10, 10)

all combinations validation result: ✓ all match
conclusion: the formula for calculating the output shape of the convolution layer is correct!

============================================================
visualize the effects of different padding and stride
============================================================
visualization results saved to: exp2/results/exp2_padding_stride_visualization.png

exp2 completed!
```
