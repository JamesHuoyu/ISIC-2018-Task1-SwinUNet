# ISIC-2018-Task1-SwinUNet
build the Swin-UNet model and solve the ISIC 2018 Task 1
## Here the standard Swin Transformer network
* the standard Swin Transformer network are provided by https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
the process is provided as follows:
```plaintxt
Input Image (224x224x3)
│
├─► [Patch Embedding] → (56x56x96)
│   │
│   ├─► [Stage1] → BasicLayer x2 → (56x56x96)
│   │   │
│   │   └─► [Patch Merging] → (28x28x192)
│   │
│   ├─► [Stage2] → BasicLayer x2 → (28x28x192)
│   │   │
│   │   └─► [Patch Merging] → (14x14x384)
│   │
│   ├─► [Stage3] → BasicLayer x6 → (14x14x384)
│   │   │
│   │   └─► [Patch Merging] → (7x7x768)
│   │
│   └─► [Stage4] → BasicLayer x2 → (7x7x768)
│       │
│       └─► [Global Avg Pooling] → (1x1x768)
│           │
│           └─► [Linear Head] → (1x1x1000)
│
└─► Output Result (1000 classes)
```
**For our network, we only use the process before the [Global Avg Pooling] and get outputs as x = self.encoder(x) after [stage4], which is already finished**
## Here the standard SwinUNet network
* the standard Swin-UNet network details are provided by https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/swin_transformer_unet_skip_expand_decoder_sys.py
* https://bgithub.xyz/HuCaoFighting/Swin-Unet/blob/main/networks/vision_transformer.py provided the Swin-UNet network api.
the process is provided as follows:
```plaintxt
Input Image (224x224x3)
│
├─► [Patch Embedding] → (56x56x96)
│   │
│   ├─► [Stage1] → BasicLayer x2 → (56x56x96)
│   │   │
│   │   └─► [Patch Merging] → (28x28x192) → save as Skip1
│   │
│   ├─► [Stage2] → BasicLayer x2 → (28x28x192)
│   │   │
│   │   └─► [Patch Merging] → (14x14x384) → save as Skip2
│   │
│   ├─► [Stage3] → BasicLayer x6 → (14x14x384)
│   │   │
│   │   └─► [Patch Merging] → (7x7x768) → save as Skip3
│   │
│   └─► [Stage4] → BasicLayer x2 → (7x7x768) → save as Bottleneck
│
├─► [Decoder]
│   │
│   ├─► [Bottleneck] → (7x7x768)
│   │   │
│   │   ├─► [PatchExpand] → (14x14x384)
│   │   │   │
│   │   │   └─► [concat with Skip3] → (14x14x768) → Linear → (14x14x384)
│   │   │       │
│   │   │       └─► BasicLayer_up x2 → (14x14x384)
│   │
│   ├─► [Stage3 decoder] → (14x14x384)
│   │   │
│   │   ├─► [PatchExpand] → (28x28x192)
│   │   │   │
│   │   │   └─► [concat with Skip2] → (28x28x384) → Linear → (28x28x192)
│   │   │       │
│   │   │       └─► BasicLayer_up x2 → (28x28x192)
│   │
│   ├─► [Stage2 decoder] → (28x28x192)
│   │   │
│   │   ├─► [PatchExpand] → (56x56x96)
│   │   │   │
│   │   │   └─► [concat with Skip1] → (56x56x192) → Linear → (56x56x96)
│   │   │       │
│   │   │       └─► BasicLayer_up x2 → (56x56x96)
│   │
│   └─► [Stage1 decoder] → (56x56x96)
│       │
│       └─► [FinalPatchExpand_X4] → (224x224x96)
│           │
│           └─► [Output Conv] → (224x224xnum_classes)
│
└─► Ouput the Segmentation Image (224x224xnum_classes)
```
## Here our trail model development process now
```plaintxt
Input Image (224x224x3)
│
├─► [Swin-Tiny encoder]
│   │
│   ├─► Stage1 → (56x56x96) → save as Skip3
│   │
│   ├─► Stage2 → (28x28x192) → save as Skip2
│   │
│   ├─► Stage3 → (14x14x384) → save as Skip1
│   │
│   └─► Stage4 → (7x7x768) → as the initial input of the decoder
│
├─► [decoder]
│   │
│   ├─► Initial input: (7x7x768)
│   │
│   ├─► Up_Sampling1 (PatchExpand) → (14x14x384)
│   │   ├─► concat with Skip1 (14x14x384) → (14x14x768)
│   │   └─► Swin Block process → (14x14x384)
│   │
│   ├─► Up_Sampling2 (PatchExpand) → (28x28x192)
│   │   ├─► concat with Skip2 (28x28x192) → (28x28x384)
│   │   └─► Swin Block process → (28x28x192)
│   │
│   ├─► Up_Sampling3 (PatchExpand) → (56x56x96)
│   │   ├─► concat with Skip3 (56x56x96) → (56x56x192)
│   │   └─► Swin Block process → (56x56x96)
│   │
│   └─► Final Up_Sampling → (224x224x1) → Sigmoid Output
│
└─► Output Segmentation (224x224)
```
### Explaining our demo model
* For encoder, we only **unfreeze** the Stage3 and Stage4 now
```python
if "layers.2" not in name and "layers.3" not in name:
    param.requires_grad = False
```
* For **Up_Sampling** in decoder:
```python
class PatchExpand:
    def forward(self, x):
        x = self.proj(x)  # (B, L, output_dim*4)
        x = x.view(B, H, W, 2, 2, self.output_dim)
        x = x.permute(0, 1, 3, 2, 4, 5) # Up Sampling 2x
```
* **Concat with skip** which in this demo is really different from the standard model, we only use the **bilinear** now, which may lack some information. So it needs more work.
```python
skip_ = F.interpolate(skip_, size=(H_x, W_x), mode='bilinear')
x = torch.cat([x, skip], dim=-1)  # channels concating
```
* **Output Setting** which is also very different from the standard model, it needs more work.
```python
x = F.interpolate(x, size=(224, 224), mode='bilinear')  # Final Up_Sampling
x = self.final_proj(x)  # with 1x1 Convj
```
