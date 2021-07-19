import tvm
import tvm.testing
from tvm import te


def split_from_inner(stage, axis, factors):
    """Split an axis by a list of factors in a reverse order
    """
    axes = []
    for f in reversed(factors):
        axis, x = stage.split(axis, f)
        axes.append(x)
    return list(reversed(axes+[axis]))

# Save into the d2ltvm package.


def bind_thread(stage, axes, tags):
    """Bind a list of axes to thread axes
    """
    for axis, tag in zip(axes, tags):
        stage.bind(axis, te.thread_axis(tag))


def get_source_code(func, target: str = "cuda"):
    if target == "cuda":
        dev_module = func.imported_modules[0]
        src_codes = dev_module.get_source()
    else:
        src_codes = func.get_source()
    return src_codes
