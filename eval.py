import timm
import time
import torch
import argparse
from ptflops import get_model_complexity_info

def benchmark(args):
    ## check model list
    # model_list = timm.list_models('convnext*')
    # [print(model) for model in model_list]

    # model_list = timm.list_models('cs*')
    # [print(model) for model in model_list]

    ## create model
    model_name = 'cspdarknet53'
    model = timm.create_model(model_name)
    model.to(args.device)
    input_size = model.default_cfg['input_size']
    print(f"Model: %s" % model_name)
    print(f"Input size: %s" % (input_size,))

    ## calculate params, flops
    ## https://github.com/sovrasov/flops-counter.pytorch
    macs, params = get_model_complexity_info(model, input_size, as_strings=True,
                                            print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    ## calculate fps
    dummy_input = torch.randn(input_size, device=args.device)
    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    times = []
    for _ in range(parser.iter):
        starting_time = time.time()
        results = model(dummy_input)
        times.append((time.time()-starting_time)*1000)
    avg_time = sum(times) / len(times)
    fps = 1000 / avg_time
    print(f"Average prediction time: {avg_time:.1f} ms ({fps:.3f} fps)")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--iter', type=int, default=300)

    benchmark(parser.parse_args())