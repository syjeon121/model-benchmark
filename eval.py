import timm
import time
import torch
import argparse
from ptflops import get_model_complexity_info

def benchmark(args):
    ## add model list
    model_list = timm.list_models('convnext_nano*')
    # [print(model) for model in model_list]

    model_table = []
    for model_name in model_list:

        ## create model
        model_info = []
        model_info.append(model_name)
        model = timm.create_model(model_name)
        model.to(args.device)
        input_size = model.default_cfg['input_size']
        model_info.append(input_size)
        # print(f"Model: %s" % model_name)
        # print(f"Input size: %s" % (input_size,))

        ## save model (onnx)
        dummy_input = torch.randn(input_size, device=args.device)
        dummy_input = torch.unsqueeze(dummy_input, dim=0)
        if args.save == True:
            # torch.save(model.state_dict(), model_name + '.pt')

            ## export the model 
            ## https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
            torch.onnx.export(model,         # model being run
                  dummy_input,               # model input (or a tuple for multiple inputs)
                  model_name + ".onnx",      # where to save the model (can be a file or file-like object)
                  export_params=False,       # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output']) # the model's output names


        ## calculate params, flops
        ## https://github.com/sovrasov/flops-counter.pytorch
        macs, params = get_model_complexity_info(model, input_size, as_strings=True,
                                                print_per_layer_stat=False, verbose=False)
        model_info.append(params)
        model_info.append(macs)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        ## calculate fps
        times = []
        for _ in range(args.iter):
            starting_time = time.time()
            results = model(dummy_input)
            times.append((time.time()-starting_time)*1000)
        avg_time = sum(times) / len(times)
        fps = round(1000 / avg_time, 1)
        model_info.append(fps)
        # print(f"Average prediction time: {avg_time:.1f} ms ({fps:.3f} fps)")

        model_table.append(model_info)

    ## sort by fps    
    model_table.sort(key = lambda x: x[-1], reverse=True)

    print("{:<30} {:<16} {:<10} {:<11} {:<10}".format('Model','Resolution','Params','GFLOPs', 'FPS'))
    for model_info in model_table:
        print("{:<30} {:<16} {:<10} {:<11} {:<10}".format(model_info[0], str(model_info[1]), model_info[2], model_info[3], model_info[4]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--iter', type=int, default=300)
    parser.add_argument('--save', default=False, action="store_true")

    benchmark(parser.parse_args())