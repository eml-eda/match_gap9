import os
import argparse
import pathlib
import match
from mako.template import Template
import numpy as np

from match_gap.utils import gap_get_result,gap_run_on_background

def c_friendly_npvalue(arr):
    # params: arr is expected to be a numpy version of the value, it should be an array but it may be also just a single value
    if len(arr.shape)>0:
        # this is actually an array and not a single value
        arr=arr.reshape([arr.shape[0]]).astype(np.uint8)
        return f'{{{str(list(arr))[1:len(str(list(arr)))-1]}}}'
    else:
        return str(arr)
            
def run_with(input_type="onnx",relay_mod=None, relay_params=None, filename=None, 
                  params_filename=None, output_path="./match_output",verbose:bool=False,
                  compare_x86:bool=True,cluster_active:bool=True,accelerator_active:bool=True,
                  single_core:bool=False,board:bool=False,
                  gap_sdk_path="/home/gap_sdk_private/"):
    
    pathlib.Path(output_path).mkdir(parents=True,exist_ok=True)
    pathlib.Path(output_path+"/src").mkdir(parents=True,exist_ok=True)
    pathlib.Path(output_path+"/include").mkdir(parents=True,exist_ok=True)
    np.random.seed(0)
    if compare_x86:

        x86_result=match.x86_run_match(input_type=input_type,relay_mod=relay_mod,relay_params=relay_params,
                                       filename=filename,params_filename=params_filename,output_path=pathlib.Path(output_path+"_x86"),
                                       keep_result=True)
        
        if verbose:
            print("\n\nx86 result:")
            print(x86_result)
            
    target=match.target.Gap9()
    target.disabled_exec_modules=[]
    if not accelerator_active:
        target.disable_exec_module("NE16")
    if not cluster_active:
        target.disable_exec_module("cluster")
    target.exec_modules_dict["NE16"].module_options=dict()
    if single_core or board:
        target.exec_modules_dict["NE16"].add_option_to_module("MATCH_NE16_RUN_SINGLE_CORE",1)
    
    res=match.match(input_type=input_type,relay_mod=relay_mod,relay_params=relay_params,
                    filename=filename,params_filename=params_filename,
                    target=target,output_path=output_path)
    
    main_code_template=Template(filename=str(pathlib.Path(os.path.dirname(__file__)))+"/gap9_lib/ones_input_template.c")
    template_data_dict=res.__dict__
    template_data_dict["target"]="gap9"
    template_data_dict["compare_with_correct"]=compare_x86
    template_data_dict["log_output"]=False
    if compare_x86:
        if len(res.match_output["shape"])==4:
            template_data_dict["expected_results"]=c_friendly_npvalue(np.asarray(x86_result["output"]).reshape(int(res.match_output["shape"][1]),int(res.match_output["shape"][2]),int(res.match_output["shape"][3])).transpose(1,2,0).flatten().astype(np.uint8))
        else:
            template_data_dict["expected_results"]=c_friendly_npvalue(np.asarray(x86_result["output"]))
    main_code=main_code_template.render(**template_data_dict)
    with open(pathlib.Path(output_path)/"src/main.c","w") as main_file:
        main_file.write(main_code)
    gap_result=gap_get_result(pathlib.Path(output_path),verbose=verbose,keep_result=True,board=board,gap_sdk_path=gap_sdk_path)
    if verbose:
        print("Gap result:")
        print(gap_result)
    
    return gap_result


def network_at(match_res,network_path,inputs=None,golden_out=None,board:bool=False,run:bool=True,gap_sdk_path="/gap_sdk"):
    main_code_template=Template(filename=str(pathlib.Path(os.path.dirname(__file__)))+f"/gap9_lib/{'fixed' if inputs is not None else 'uart'}_input_template.c")
    template_data_dict=match_res.__dict__
    template_data_dict["target"]="gap9"
    template_data_dict["compare_with_correct"]=golden_out is not None
    template_data_dict["log_output"]=True
    if inputs is not None:
        template_data_dict["inputs"]=[{"c_arr_size":input_["size"],"c_arr_values":c_friendly_npvalue(np.asarray(input_["values"])),**input_} for input_ in inputs]

    main_code=main_code_template.render(**template_data_dict)
    with open(pathlib.Path(network_path)/"src/main.c","w") as main_file:
        main_file.write(main_code)

    gap_result=gap_get_result(pathlib.Path(network_path),verbose=False,keep_result=True,board=board,gap_sdk_path=gap_sdk_path,run=run,clean=False)
    
    return gap_result

def uart_network(match_res,network_path,gap_sdk_path="/gap_sdk",board:bool=False):
    network_at(match_res=match_res,network_path=network_path,inputs=None,golden_out=None,board=board,run=False,gap_sdk_path=gap_sdk_path)

    gap_run_on_background(pathlib.Path(network_path),gap_sdk_path=gap_sdk_path,board=board)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="increase log verbosity"
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_const",
        dest="verbose",
        const=2,
        help="log debug messages (same as -vv)",
    )

    parser.add_argument(
        "-i",
        "--input",
        dest="input_type",
        type=str,
        help="Type of the input to compile, possible values are [onnx,relay].",
    )

    parser.add_argument(
        "-f",
        "--filename",
        dest="filename",
        type=str,
        help="Provide the absolute filename of the module to compile.",
    )

    parser.add_argument(
        "-p",
        "--params",
        dest="params_filename",
        type=str,
        help="Provide the filename of the params needed by the module to compile.",
    )

    parser.add_argument(
        "-c",
        "--convexample",
        dest="convexample",
        action="store_true",
        help="compile a simple 2d convolution example, that contains a con2d, a bias add and a requantization step",
    )

    parser.add_argument(
        "-a",
        "--addexample",
        dest="addexample",
        action="store_true",
        help="compile a simple add example between 2 2d convs like the ones in the convexample,"
        + "with a final requantization step",
    )

    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        default="./match_output",
        type=str,
        help="Provide the absolute output path"
    )

    parser.add_argument(
        "--ne16",
        dest="ne16",
        action="store_true",
        help="Deploy considering the accelerator"
    )

    parser.add_argument(
        "--cluster",
        dest="cluster",
        action="store_true",
        help="Deploy considering the cluster"
    )

    parser.add_argument(
        "-x",
        "--x86",
        dest="x86",
        action="store_true",
        help="Compare the result with the x86 one"
    )

    parser.add_argument(
        "--board",
        dest="board",
        action="store_true",
        help="Deploy on the board(defaults to use gvsoc)"
    )

    parser.add_argument(
        "-s",
        "--single_core",
        dest="single_core",
        action="store_true",
        help="Run NE16 with a single cluster core"
    )

    parser.add_argument(
        "-g",
        "--gap_sdk",
        dest="gap_sdk_path",
        default="/home/gap_sdk_private/",
        type=str,
        help="Provide the absolute path to the GAP SDK, defaults to /home/gap_sdk/private"
    )
    args = parser.parse_args()
    input_type=args.input_type
    mod=None
    params=None
    filename=args.filename
    params_filename=args.params_filename
    output_path=args.output_path
    compare_x86=args.x86
    single_core=args.single_core
    board=args.board
    gap_sdk_path=args.gap_sdk_path

    if args.convexample:
        mod,params=match.create_model_conv_2d()
    elif args.addexample:
        mod,params=match.create_model_add_convs()

    res_=run_with(
        input_type=input_type,
        relay_mod=mod,
        relay_params=params,
        filename=filename,
        params_filename=params_filename,
        output_path=output_path,
        compare_x86=compare_x86,
        cluster_active=args.cluster,
        accelerator_active=args.ne16,
        single_core=single_core,
        board=board,
        gap_sdk_path=gap_sdk_path,
    )

    if compare_x86:
        print(f'Result is {""if res_["correct"] else "NOT "}correct')
    else:
        print("Result is",res_)