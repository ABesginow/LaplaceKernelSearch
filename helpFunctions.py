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
    elif kernel_expression._get_name() == "MaternKernel":
        return "MAT"
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


def generate_kernel_from_polish_string(s: str):
    from gpytorch.kernels import ScaleKernel, AdditiveKernel, ProductKernel, RBFKernel, PeriodicKernel, LinearKernel
    def get_corresponding_kernel(list_of_elements):
        try:
            name = list_of_elements.pop(0)
        except:
            return None
        match name:
            case "*":
                return ProductKernel(get_corresponding_kernel(list_of_elements), get_corresponding_kernel(list_of_elements))
            case "+":
                return AdditiveKernel(get_corresponding_kernel(list_of_elements), get_corresponding_kernel(list_of_elements))
            case "c":
                return ScaleKernel(get_corresponding_kernel(list_of_elements))
            case "SE":
                return RBFKernel()
            case "RBF":
                return RBFKernel()
            case "PER":
                return PeriodicKernel()
            case "LIN":
                return LinearKernel()
    s2 = s.split(' ')
    return get_corresponding_kernel(s2)