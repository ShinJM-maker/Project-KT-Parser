from .base import BaseMetric, LabelRequiredMetric
from .functional import (
    klue_dp_las_macro_f1,
    klue_dp_las_micro_f1,
    klue_dp_uas_macro_f1,
    klue_dp_uas_micro_f1,
)


KlueDP_UASMacroF1 = BaseMetric(klue_dp_uas_macro_f1)
KlueDP_UASMicroF1 = BaseMetric(klue_dp_uas_micro_f1)
KlueDP_LASMacroF1 = BaseMetric(klue_dp_las_macro_f1)
KlueDP_LASMicroF1 = BaseMetric(klue_dp_las_micro_f1)