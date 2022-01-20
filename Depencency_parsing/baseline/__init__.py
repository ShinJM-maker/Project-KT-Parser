""" klue_baseline package info """
__version__ = "0.1.0"
__author__ = "KLUE project contributors"


from baseline.data import (
    KlueDPProcessor,

)
from baseline.metrics import (
    KlueDP_LASMacroF1,
    KlueDP_LASMicroF1,
    KlueDP_UASMacroF1,
    KlueDP_UASMicroF1,
)
from baseline.models import (
    DPTransformer,
)
from baseline.task import KlueTask

# Register Task - KlueTask(processor, model_type, metrics)
KLUE_TASKS = {
    "dp": KlueTask(
        KlueDPProcessor,
        DPTransformer,
        {
            "uas_macro_f1": KlueDP_UASMacroF1,
            "uas_micro_f1": KlueDP_UASMicroF1,
            "las_macro_f1": KlueDP_LASMacroF1,
            "las_micro_f1": KlueDP_LASMicroF1,
        },
    ),
}
