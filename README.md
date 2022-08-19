# model-benchmark
this repo is a simple code for evaluating pytorch [timm](https://github.com/rwightman/pytorch-image-models) models (params, flops, fps)
this repo used [ptflops](https://github.com/sovrasov/flops-counter.pytorch)

## bechmark

CPU: Intel(R) Core(TM) i5-1038NG7 CPU @ 2.00GHz CPU and no GPU.

| Model     | Resolution | Params(M) | GFLOPs | FPS |
| :-------- | :--------: | :--------: | :----: | :---------:|
| convnext_nano | (3, 224, 224) | 15.59 | 2.46 | 16.8 |
| convnext_nano_hnf | (3, 224, 224) | 15.59 | 2.46 | 16.9 |
| convnext_nano_ols | (3, 224, 224) | 15.61 | 2.51 | 16.2 |

## example

python3 eval.py --device cpu --iter 20
```
Model                          Resolution       Params     GFLOPs      FPS       
convnext_nano                  [(3, 224, 224), '15.59 M', '2.46 GMac', 16.8]
convnext_nano_hnf              [(3, 224, 224), '15.59 M', '2.46 GMac', 16.9]
convnext_nano_ols              [(3, 224, 224), '15.61 M', '2.51 GMac', 16.2]
```