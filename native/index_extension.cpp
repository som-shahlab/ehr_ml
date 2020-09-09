#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "reader.h"

namespace py = pybind11;

void register_index_extension(py::module& root) {
    py::module m = root.def_submodule("index");
    py::class_<Index>(m, "Index")
        .def(py::init<const char*>(), py::arg("filename"))
        .def("get_patient_ids", &Index::get_patient_ids, py::arg("term"))
        .def("get_all_patient_ids", &Index::get_all_patient_ids,
             py::arg("terms"));
}