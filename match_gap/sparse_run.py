import argparse
import numpy as np
import os, sys, pathlib
import mako.template as tpl
MATCH_OUTPUT = f'/mnt/c/Users/aless/Desktop/Alessio-Work/Repositories/match_gap9/match_gap/match_output'
GAP_SDK = pathlib.Path(f'/home/burry94/gap9-sdk-sparse/')
os.environ["MATCH_PATH"]=f'/mnt/c/Users/aless/Desktop/Alessio-Work/Repositories/match'
os.environ["TVM_HOME"]=os.environ["MATCH_PATH"]+"/match-tvm"
# export PYTHONPATH="${PYTHONPATH}:/mnt/c/Users/aless/Desktop/Alessio-Work/Repositories/match/match-tvm/python"
sys.path.insert(0,f"{os.environ.get('TVM_HOME')}/python")

# export PYTHONPATH="${PYTHONPATH}:/mnt/c/Users/aless/Desktop/Alessio-Work/Repositories/match/zigzag"
sys.path.insert(0,f"{os.environ.get('MATCH_PATH')}/zigzag")
import relay_resnet_sparse as sparse_relay 
import match
from utils import gap_get_result

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # example to run the old script
    # python3 match_gap/run.py -o MATCH_OUTPUT --cluster -i onnx -f ../match/examples/timeppg_small.onnx -g /home/burry/gap_sdk_private -v

    parser.add_argument(
        "--only-run",
        type=str
    )
    pathlib.Path(MATCH_OUTPUT).mkdir(parents=True,exist_ok=True)
    pathlib.Path(MATCH_OUTPUT+"/src").mkdir(parents=True,exist_ok=True)
    pathlib.Path(MATCH_OUTPUT+"/include").mkdir(parents=True,exist_ok=True)
    args = parser.parse_args()
    if args.only_run == "T":
        gap_result=gap_get_result(pathlib.Path(MATCH_OUTPUT),verbose=False,keep_result=True,board=False,gap_sdk_path=GAP_SDK)
        exit(0) 
    np.random.seed(0)
    out = sparse_relay.create_custom_layers(weight_bits=8, add_layout_transforms=False, mixed=False)   
    target=match.target.Gap9()
    target.disabled_exec_modules=[]
    target.disable_exec_module("NE16")
    target.exec_modules_dict["NE16"].module_options=dict()
    res=match.match(input_type='relay',relay_mod=out['model'],relay_params=out['params'],
                    filename=None,params_filename=None,
                    target=target,output_path=MATCH_OUTPUT)
    
    main_code_template = tpl.Template(filename=str(pathlib.Path(os.path.dirname(__file__)))+"/gap9_lib/ones_input_template.c")
    template_data_dict=res.__dict__
    template_data_dict["target"]="gap9"
    template_data_dict["compare_with_correct"]=False
    template_data_dict["log_output"]=False
    main_code=main_code_template.render(**template_data_dict)
    with open(pathlib.Path(MATCH_OUTPUT)/"src/main.c","w") as main_file:
        main_file.write(main_code)
    gap_result=gap_get_result(pathlib.Path(MATCH_OUTPUT),verbose=False,keep_result=True,board=False,gap_sdk_path=GAP_SDK)
