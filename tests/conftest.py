import os
import sys

if sys.platform == "win32":
    user_torch_lib = os.path.expanduser(r"~\AppData\Roaming\Python\Python311\site-packages\torch\lib")
    if os.path.exists(user_torch_lib):
        try:
            os.add_dll_directory(user_torch_lib)
            os.environ["PATH"] = user_torch_lib + os.pathsep + os.environ.get("PATH", "")
        except (OSError, ImportError):
            pass

try:
    import torch  # noqa: F401
except ImportError:
    pass
