# ISIC-2018-Task1-SwinUNet
build the Swin-UNet model and solve the ISIC 2018 Task 1
## Here the standard SwinUNet network
the standard Swin-UNet network details are provided by https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/swin_transformer_unet_skip_expand_decoder_sys.py
and https://bgithub.xyz/HuCaoFighting/Swin-Unet/blob/main/networks/vision_transformer.py provided the Swin-UNet network api.
the process is provided as follows:
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
│   │   │   └─► [skiped with Skip3] → (14x14x768) → Linear → (14x14x384)
│   │   │       │
│   │   │       └─► BasicLayer_up x2 → (14x14x384)
│   │
│   ├─► [Stage3 decoder] → (14x14x384)
│   │   │
│   │   ├─► [PatchExpand] → (28x28x192)
│   │   │   │
│   │   │   └─► [skiped with Skip2] → (28x28x384) → Linear → (28x28x192)
│   │   │       │
│   │   │       └─► BasicLayer_up x2 → (28x28x192)
│   │
│   ├─► [Stage2 decoder] → (28x28x192)
│   │   │
│   │   ├─► [PatchExpand] → (56x56x96)
│   │   │   │
│   │   │   └─► [skiped with Skip1] → (56x56x192) → Linear → (56x56x96)
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
