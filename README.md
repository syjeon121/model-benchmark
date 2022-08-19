# model-benchmark
this repo is a simple code for evaluating pytorch model (params, flops, fps)
this repo used [ptflops](https://github.com/sovrasov/flops-counter.pytorch)

## bechmark

CPU: Intel(R) Core(TM) i5-1038NG7 CPU @ 2.00GHz CPU and no GPU.

| Model     | resolution | #Params(M) | GFLOPs | FPS |
| :-------- | :--------: | :--------: | :----: | :---------:|
| convnext_base | 224 | 88.5 | 15.4 | 4.2 |
| cspdarknet53 | 256 | 27.6 | 6.5 | 7.8 |

## example

python3 eval.py
```
Model: cspdarknet53
Input size: (3, 256, 256)
Computational complexity:       6.55 GMac
Number of parameters:           27.64 M 
Average prediction time: 128.6 ms (7.775 fps)
```