#include "ActiveRotatingFilter.h"
#include "RotationInvariantEncoding.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("arf_mappingrotate_forward", &ARF_MappingRotate_forward, "active rotating filter forward");
  m.def("arf_mappingrotate_backward", &ARF_MappingRotate_backward, "active rotating filter backward");
  m.def("rie_alignfeature_forward", &RIE_AlignFeature_forward, "rotation invariant encoding forward");
  m.def("rie_alignfeature_backward", &RIE_AlignFeature_backward, "rotation invariant encoding backward");
}
