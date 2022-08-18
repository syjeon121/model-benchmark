# model-benchmark
benchmark pytorch models
ref: https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055
https://github.com/nebuly-ai/nebullvm/blob/main/resources/notebooks/Accelerate-PyTorch-YOLO-with-nebullvm.ipynb
https://github.com/sovrasov/flops-counter.pytorch

### bechmark 1

CPU: Intel(R) Core(TM) i5-1038NG7 CPU @ 2.00GHz CPU and no GPU.

| Model     | #Params(M) | GFLOPs | FPS |
| :-------- | :--------: | :----: | :---------:|
|   | | | |

### bechmark 2


GPU: Titan XP  
CPU: AMD Ryzen Threadripper 1950X 16-Core Processor

| Model     | #Params(M) | GFLOPs | FPS |
| :-------- | :--------: | :----: | :---------:|
| [VAN-base](https://github.com/Visual-Attention-Network/VAN-Classification)  |    26.6    |  5.0  |    76.5     |
| [centernext - convnext-t](https://github.com/MarkAny-Vision-AI/CenterNeXt)  |    31.7    |  5.1  |    151.7    |
