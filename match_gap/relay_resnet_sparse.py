from match.relay.utils.utils import (
    relay_gap9_conv2d,
    relay_gap9_dense,
    create_random_array,
    relay_gap9_add,
)
from match.relay.utils.utils import numpy_to_array
import numpy as np
import tvm.relay as relay
import tvm

M = 16
N = 1
NUM_CLASSES = 100  # CIFAR100


def create_random_array_dense(shape, dtype):
    """
    WARNING: This function MUST receive:
    - a dense tensor with KCHW or (Neurons_out, Neurons_in) format.
    """
    output_array_tvm: tvm.runtime.ndarray.NDArray = create_random_array(shape, dtype)
    return output_array_tvm
    
def create_random_array_sparse(shape, dtype):
    """
    WARNING: This function MUST receive:
    - a dense tensor with KCHW or (Neurons_out, Neurons_in) format.
    - N:M variables to be used.
    """
    output_array_tvm: tvm.runtime.ndarray.NDArray = create_random_array(shape, dtype)
    output_array = output_array_tvm.numpy()
    is_compatible = ((output_array.size / output_array.shape[0]) % M) == 0
    if not is_compatible:
        print(f"Conv {shape} not compatible with M = {M}")
        return output_array_tvm

    groups = int(output_array.size / M)

    if len(shape) == 4:
        kchw = output_array.shape
        khwc = (kchw[0], kchw[2], kchw[3], kchw[1])
        to_chw = (0, 3, 1, 2)
        K = output_array.shape[0]
        # KCHW -> KHWC format
        tmp = output_array.transpose(0, 2, 3, 1).reshape(groups, M)
    else:
        kchw = output_array.shape
        khwc = (kchw[0], kchw[1])
        to_chw = (0, 1)
        tmp = output_array.reshape(groups, M)

    tmp = np.abs(tmp)
    mask = np.argsort(tmp, axis=1)[:, : int(M - N)]
    # Binary mask
    tmp_o = np.ones(tmp.shape)
    np.put_along_axis(tmp_o, mask, axis=1, values=0)
    tmp_o = tmp_o.reshape(khwc)
    tmp_o = np.transpose(tmp_o, to_chw)
    tmp = output_array * (tmp_o)
    tmp = tmp.astype(dtype)
    return numpy_to_array(tmp, dtype)


def basic_block(y, in_channels, out_channels, stride, i=0):
    params = {}

    weights_shape = (out_channels, in_channels, 3, 3)
    weights = create_random_array_sparse(weights_shape, f"int8")
    bias = create_random_array(weights_shape[0], "int32")
    x, params_conv1 = relay_gap9_conv2d(
        y,
        f"conv{i}",
        weights,
        bias,
        strides=(stride, stride),
        padding=(1, 1),
        act=True,
        shift_bits=4,
        batchnorm=True,
    )

    weights_shape = (out_channels, out_channels, 3, 3)
    weights = create_random_array_sparse(weights_shape, f"int8")
    bias = create_random_array(weights_shape[0], "int32")
    x, params_conv2 = relay_gap9_conv2d(
        x,
        f"conv{i+1}",
        weights,
        bias,
        strides=(1, 1),
        padding=(1, 1),
        act=False,
        shift_bits=4,
        batchnorm=True,
    )
    i += 2
    params_conv1.update(params_conv2)

    if stride != 1 or in_channels != out_channels:
        weights_shape = (out_channels, in_channels, 1, 1)
        weights = create_random_array_sparse(weights_shape, f"int8")
        bias = create_random_array(weights_shape[0], "int32")
        y, params_conv3 = relay_gap9_conv2d(
            y,
            f"conv_pw{i+1}",
            weights,
            bias,
            strides=(stride, stride),
            padding=(0, 0),
            act=False,
            shift_bits=4,
            batchnorm=True,
        )
        params_conv1.update(params_conv3)
    x = relay_gap9_add(x, y, f"add{i+1}")
    params = params_conv1
    return i, x, params


def create_model_resnet18(
    weight_bits: int = 8, add_layout_transforms: bool = False, mixed: bool = False
):
    full = False
    if full:
        input_shape = (1, 4, 32, 32)
    else:
        input_shape = (1, 64, 32, 32)
    num_classes = 10
    # Using input_0 to be used with create_demo_file
    x = relay.var("input_0", relay.TensorType(input_shape, "int8"))

    num_filters = 64

    # Input sequential
    weights_shape = (64, 64, 3, 3)
    # weights = create_random_array_dense(weights_shape, f"int8")
    weights = create_random_array_sparse(weights_shape, f"int8")
    bias = create_random_array(weights_shape[0], "int32")
    x, params_conv1 = relay_gap9_conv2d(
        x,
        f"conv0",
        weights,
        bias,
        strides=(1, 1),
        padding=(1, 1),
        act=True,
        shift_bits=4,
        batchnorm=True,
    )

    # Conv2_x
    if full:
        i, x, params_block1 = basic_block(x, in_channels=64, out_channels=64, stride=1, i=1)
        i, x, params_block2 = basic_block(x, in_channels=64, out_channels=64, stride=1, i=i)
        # Conv3_x
        i, x, params_block3 = basic_block(
            x, in_channels=64, out_channels=128, stride=2, i=i
        )
        i, x, params_block4 = basic_block(
            x, in_channels=128, out_channels=128, stride=1, i=i
        )

        # Conv4_x
        i, x, params_block5 = basic_block(
            x, in_channels=128, out_channels=256, stride=2, i=i
        )
        i, x, params_block6 = basic_block(
            x, in_channels=256, out_channels=256, stride=1, i=i
        )

        # Conv5_x
        i, x, params_block7 = basic_block(
            x, in_channels=256, out_channels=512, stride=2, i=i
        )
        i, x, params_block8 = basic_block(
            x, in_channels=512, out_channels=512, stride=1, i=i
        )

        # x = relay.nn.avg_pool2d(x, (4, 4))
        # x = relay.reshape(x, (512, NUM_CLASSES))

        # weights_shape = (num_classes, num_filters)
        # weights = create_random_array(weights_shape, "int8")
        # bias = create_random_array(weights_shape[0], "int32")
        # x, params_dense = relay_gap9_dense(
        #     x, "dense", weights, bias, act=False, shift_bits=4, batchnorm=False
        # )

        # x = relay.cast(
        #     x, "float32"
        # )  # cast needed since softmax in TVM seems not to work with integer inputs
        # x = relay.op.nn.softmax(x)

    if full:
        params_conv1.update(params_block1)
        params_conv1.update(params_block2)
        params_conv1.update(params_block3)
        params_conv1.update(params_block4)
        params_conv1.update(params_block5)
        params_conv1.update(params_block6)
        params_conv1.update(params_block7)
        params_conv1.update(params_block8)

        # params_conv1.update(params_dense)
    params = params_conv1

    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return {"model": mod, "params": params, "shape": input_shape}


def create_custom_layers(
    weight_bits: int = 8, add_layout_transforms: bool = False, mixed: bool = False
):
    input_shape = (1, 64, 32, 32)
    # Using input_0 to be used with create_demo_file
    x = relay.var("input_0", relay.TensorType(input_shape, "int8"))
    # Input sequential
    weights_shape = (64, 64, 3, 3)
    weights_shape1 = (128, 64, 3, 3)
    # weights = create_random_array_dense(weights_shape, f"int8")
    weights = create_random_array_sparse(weights_shape, f"int8")
    bias = create_random_array(weights_shape[0], "int32")
    x, params_conv1 = relay_gap9_conv2d(
        x,
        f"conv0",
        weights,
        bias,
        strides=(1, 1),
        padding=(1, 1),
        act=True,
        shift_bits=4,
        batchnorm=True,
    )
    weights1 = create_random_array_sparse(weights_shape1, f"int8")
    bias1 = create_random_array(weights_shape1[0], "int32")
    x, params_conv2 = relay_gap9_conv2d(
        x,
        f"conv1",
        weights1,
        bias1,
        strides=(1, 1),
        padding=(1, 1),
        act=True,
        shift_bits=4,
        batchnorm=True,
    )
    params_conv1.update(params_conv2)

        # params_conv1.update(params_dense)
    params = params_conv1

    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return {"model": mod, "params": params, "shape": input_shape}