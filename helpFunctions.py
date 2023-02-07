import defaultOptions

def get_kernels_in_kernel_expression(kernel_expression):
    """
    returns list of all base kernels in a kernel expression
    """
    if kernel_expression == None:
        return []
    if hasattr(kernel_expression, "kernels"):
        ret = list()
        for kernel in kernel_expression.kernels:
            ret.extend(get_kernels_in_kernel_expression(kernel))
        return ret
    elif kernel_expression._get_name() == "ScaleKernel":
        return get_kernels_in_kernel_expression(kernel_expression.base_kernel)
    elif kernel_expression._get_name() == "GridKernel":
        return get_kernels_in_kernel_expression(kernel_expression.base_kernel)
    else:
        return [kernel_expression]

def print_formatted_hyperparameters(kernel_expression):
    full_string = ""
    for kernel in get_kernels_in_kernel_expression(kernel_expression):
        full_string += kernel._get_name()
        for name, param, constraint in kernel.named_parameters_and_constraints():
            full_string += f"\t{name[name.rfind('.raw')+5:]:13}: {constraint.transform(param.data).item()}"
    return full_string



def get_string_representation_of_kernel(kernel_expression):
    if kernel_expression._get_name() == "AdditiveKernel":
        s = ""
        for k in kernel_expression.kernels:
            s += get_string_representation_of_kernel(k) + " + "
        return "(" + s[:-3] + ")"
    elif kernel_expression._get_name() == "ProductKernel":
        s = ""
        for k in kernel_expression.kernels:
            s += get_string_representation_of_kernel(k) + " * "
        return "(" + s[:-3] + ")"
    elif kernel_expression._get_name() == "ScaleKernel":
        return f"(c * {get_string_representation_of_kernel(kernel_expression.base_kernel)})"
    elif kernel_expression._get_name() == "RBFKernel":
        return "SE"
    elif kernel_expression._get_name() == "LinearKernel":
        return "LIN"
    elif kernel_expression._get_name() == "PeriodicKernel":
        return "PER"
    else:
        return kernel_expression._get_name()


def unraw_parameter_names(generator):
    for param_name, param in generator:
        if ".raw_" not in param_name:
            yield param_name
        else:
            yield param_name.replace(".raw_", ".")

def depth_of_kernel(kernel):
    if not hasattr(kernel, "kernels"):
        return 1
    else:
        return 1 + max([depth_of_kernel(k) for k in kernel.kernels])

def amount_of_base_kernels(kernel):
    if hasattr(kernel, "base_kernel"):
        return amount_of_base_kernels(kernel.base_kernel)
    elif not hasattr(kernel, "kernels"):
        return 1
    else:
        return sum([amount_of_base_kernels(k) for k in kernel.kernels])

def kernel_contains(kernel, segment):
    """
    Returns whether or not a given kernel expression contains a subexpression as a segment, regardless of HPs
    Args:
        kernel: a kernel expression
        segment: a kernel expression

    Returns: bool
    """
    return get_string_representation_of_kernel(segment)[1:-1] in get_string_representation_of_kernel(kernel)


def clean_kernel_expression(kernel):
    # option 1: scale(scale(...))
    if kernel._get_name() == "ScaleKernel":
        if kernel.base_kernel._get_name() == "ScaleKernel":
            kernel.base_kernel = kernel.base_kernel.base_kernel
    elif hasattr(kernel, "kernels"):
        for sub_expression in kernel.kernels:
            clean_kernel_expression(sub_expression)

def generate_kernel_from_polish_string(s : str):
    """
    generate a gpytorch kernel expression equivalent to the string s
    Args:
        s: a string representing the kernel in polish notation. Place spaces between elements
        Supported kernels are SE, PER, LIN, c (use as "c PER" or "c SE" for example)
        Supported operators are + and *
        Scaling operators (e.g. c + PER SE) is currently not supported

    Returns: gpytorch kernel
    """
    from gpytorch.kernels import ScaleKernel, AdditiveKernel, ProductKernel, RBFKernel, PeriodicKernel, LinearKernel
    def get_corresponding_kernel(name):
        match name:
            case "*":
                return ProductKernel()
            case "+":
                return AdditiveKernel()
            case "c":
                return ScaleKernel
            case "SE":
                return RBFKernel()
            case "RBF":
                return RBFKernel()
            case "PER":
                return PeriodicKernel()
            case "LIN":
                return LinearKernel()
    s2 = s.split(' ')
    first = s2.pop(0)
    if first == "c":
        return ScaleKernel(get_corresponding_kernel(s2.pop(0))) # special case
    kernel_expression = get_corresponding_kernel(first)
    kernel_buffer = []
    if hasattr(kernel_expression, "kernels"):
        kernel_buffer.append(kernel_expression)
    while True:
        if len(kernel_buffer) == 0: # if buffer is empty, the expression is finished
            break
        if len(kernel_buffer[-1].kernels) == 2: # if the latest operator has 2 children, it is full and gets removed
            kernel_buffer = kernel_buffer[:-1]
        else:
            try:
                current_symbol = s2.pop(0)
            except: # s2 is empty -> done
                break
            if current_symbol == "c":
                current_kernel = ScaleKernel(get_corresponding_kernel(s2.pop(0)))
            else:
                current_kernel = get_corresponding_kernel(current_symbol)
            kernel_buffer[-1].kernels.append(current_kernel)
            if hasattr(current_kernel, 'kernels'):
                kernel_buffer.append(current_kernel)
    return kernel_expression