# vncorenlp_singleton.py
import jpype
from py_vncorenlp import VnCoreNLP

_vncore_model = None

def get_vncore_model(model_dir):
    global _vncore_model
    if _vncore_model is None:
        if not jpype.isJVMStarted():
            _vncore_model = VnCoreNLP(save_dir=model_dir)
        else:
            print("[WARNING] JVM is already started. Cannot re-initialize VnCoreNLP.")
            return None
    return _vncore_model
