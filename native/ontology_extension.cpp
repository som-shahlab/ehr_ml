#include "ontology_extension.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

#include "constdb.h"
#include "reader.h"

namespace py = pybind11;

void register_ontology_extension(py::module& root) {
    py::module m = root.def_submodule("ontology");
    py::class_<OntologyReader>(m, "OntologyReader")
        .def(py::init<const char*>(), py::arg("filename") = std::nullopt)
        .def("get_subwords", &OntologyReader::get_subwords, py::arg("code"),
             py::keep_alive<0, 1>())
        .def("get_parents", &OntologyReader::get_parents, py::arg("subword"),
             py::keep_alive<0, 1>())
        .def("get_all_parents", &OntologyReader::get_all_parents,
             py::arg("subword"), py::keep_alive<0, 1>())
        .def("get_children", &OntologyReader::get_children, py::arg("subword"),
             py::keep_alive<0, 1>())
        .def("get_words_for_subword", &OntologyReader::get_words_for_subword,
             py::arg("code"), py::keep_alive<0, 1>())
        .def("get_words_for_subword_term",
             &OntologyReader::get_words_for_subword_term, py::arg("term"),
             py::keep_alive<0, 1>())
        .def("get_recorded_date_codes",
             &OntologyReader::get_recorded_date_codes, py::keep_alive<0, 1>())
        .def("get_dictionary", &OntologyReader::get_dictionary,
             py::return_value_policy::reference_internal)
        .def("get_root", &OntologyReader::get_root_code)
        .def("get_text_description_dictionary",
             &OntologyReader::get_text_description_dictionary,
             py::return_value_policy::reference_internal);
    py::class_<OntologyCodeDictionary>(m, "OntologyCodeDictionary")
        .def("get_word", &OntologyCodeDictionary::get_word, py::arg("code"))
        .def("get_definition", &OntologyCodeDictionary::get_definition,
             py::arg("code"))
        .def("map", &OntologyCodeDictionary::map, py::arg("code"));
}
