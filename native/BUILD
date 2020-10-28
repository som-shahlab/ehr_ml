load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")
load("@rules_foreign_cc//tools/build_defs:configure.bzl", "configure_make")

configure_make(
    name = "lzma",

    shared_libraries = ["liblzma.so"],
    static_libraries = ["liblzma.a"],
    lib_source = "@source_xz//:all",
)

cmake_external(
    name = "zlib_ng",
    # Values to be passed as -Dkey=value on the CMake command line;
    # here are serving to provide some CMake script configuration options
    cache_entries = {
        "WITH_NATIVE_INSTRUCTIONS": "on",
        "WITH_GZFILEOP": "on",
        "ZLIB_COMPAT": "on",
    },
    lib_source = "@source_zlib_ng//:all",

    # We are selecting the resulting static library to be passed in C/C++ provider
    # as the result of the build;
    # However, the cmake_external dependants could use other artefacts provided by the build,
    # according to their CMake script
    
    shared_libraries = ["libz.so"],
    static_libraries = ["libz.a"],
)

cmake_external(
    name = "snappy",
    lib_source = "@source_snappy//:all",

    cache_entries = {
        "BUILD_SHARED_LIBS": "on",
    },

    shared_libraries = ["libsnappy.so"],
)

cmake_external(
    name = "jansson",
    lib_source = "@source_jansson//:all",

    static_libraries = ["libjansson.a"],
)

cmake_external(
    name = "avro",
    lib_source = "@source_avro//:all",
    working_directory = "lang/c",

    cache_entries = {
        "ZLIB_LIBRARY": "$EXT_BUILD_DEPS/zlib_ng/lib/libz.so",
        "ZLIB_INCLUDE_DIR": "$EXT_BUILD_DEPS/zlib_ng/include",
        "SNAPPY_INCLUDE_DIRS": "$EXT_BUILD_DEPS/snappy/include",
        "SNAPPY_LIBRARIES": "$EXT_BUILD_DEPS/snappy/lib/libsnappy.so",
        "LZMA_INCLUDE_DIRS": "$EXT_BUILD_DEPS/lzma/include",
        "LZMA_LIBRARIES": "$EXT_BUILD_DEPS/lzma/lib/liblzma.so",
        "JANSSON_INCLUDE_DIRS": "$EXT_BUILD_DEPS/jansson/include",
        "JANSSON_LIBRARIES": "$EXT_BUILD_DEPS/jansson/lib/libjansson.a",
        "JANSSON_FOUND": "on",
        "CMAKE_SKIP_INSTALL_ALL_DEPENDENCY": "on",
    },

    deps = [
        ":zlib_ng",
        ":snappy",
        ":lzma",
        ":jansson",
    ],

    # We are selecting the resulting static library to be passed in C/C++ provider
    # as the result of the build;
    # However, the cmake_external dependants could use other artefacts provided by the build,
    # according to their CMake script
    static_libraries = ["libavro.a"],
)

cc_library(
    name = "flatmap",
    hdrs = ["flatmap.h"],
    deps = []
)

cc_library(
    name = "avro_utils",
    hdrs = ["avro_utils.h"],
    deps = [
        ":avro", 
        "@boost//:filesystem",
        "@com_google_absl//absl/time",
    ]
)

cc_binary(
    name="sort_avro",
    srcs = [
        "sort_avro.cpp",
    ],
    deps = [
        ":avro",
        "@boost//:filesystem",
        "@concurrentqueue",
    ],
    linkopts = [ "-lpthread" ]
)

cc_library (
    name="lzma_utils",
    hdrs = [
        "lzma_utils.h",
    ],
    deps = [
        ":lzma",
        "@com_google_absl//absl/strings",
    ],
)

cc_library (
    name="parse_utils",
    hdrs = [
        "parse_utils.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library (
    name="umls",
    hdrs = [
        "umls.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_test (
    name="umls_test",
    srcs = [
        "umls_test.cpp",
    ],
    deps = [
        ":umls",
        "@gtest//:gtest_main",
    ],
    linkopts = ["-lrt"],
)

cc_library (
    name="rxnorm",
    hdrs = [
        "rxnorm.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test (
    name="rxnorm_test",
    srcs = [
        "rxnorm_test.cpp",
    ],
    deps = [
        ":rxnorm",
        "@gtest//:gtest_main",
    ],
    linkopts = ["-lrt"],
)


cc_library (
    name="gem",
    hdrs = [
        "gem.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)


cc_library (
    name="csv",
    hdrs = [
        "csv.h",
    ],
    deps = [
        ":zlib_ng",
        "@com_google_absl//absl/strings",
    ],
)

cc_library (
    name="concept",
    hdrs = [
        "concept.h",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        ":avro_utils",
        "@boost//:filesystem",
        ":parse_utils",
    ],
)

cc_library (
    name="reader",
    hdrs = [
        "reader.h",
    ],
    deps = [
        ":constdb",
        "@json",
        "@streamvbyte",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
    linkopts = ["-lrt"],
)

cc_library (
    name="writer",
    hdrs = [
        "writer.h",
    ],
    deps = [
        ":constdb",
        ":reader",
    ],
)

cc_library (
    name="constdb",
    hdrs = [
        "constdb.h"
    ],
    srcs = [
        "constdb.cpp",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
    ]
)

cc_library(
    name = "python",
    hdrs = glob(["python/*.h", "python/*/*.h"]),
    includes = ["python"],
)

cc_library(
    name = "numpy",
    hdrs = glob(["numpy/numpy/*.h"]),
)

cc_binary(
    name="extract_omop",
    srcs=[
        "extract_omop.cpp"
    ],
    deps = [
        ":csv",
        ":constdb",
        ":reader",
        ":concept",
        "@streamvbyte",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ]
)

cc_binary(
    name="omop_ontology",
    srcs=[
        "omop_ontology.cpp"
    ],
    deps = [
        ":csv",
        ":constdb",
        ":reader",
        ":concept",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name="ontology_starr",
    srcs=[
        "ontology_starr.cpp"
    ],
    deps = [
        ":csv",
        ":constdb",
        ":reader",
        ":concept",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@boost//:filesystem",
        ":umls",
    ],
)

cc_binary(
    name="extract_starr",
    srcs=[
        "extract_starr.cpp"
    ],
    deps = [  
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        ":writer",
        ":parse_utils",
        "@readerwriterqueue",
        ":avro_utils",
        "@boost//:filesystem",
    ]
)


cc_binary(
    name="clean_starr",
    srcs=[
        "clean_starr.cpp"
    ],
    deps = [  
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        ":writer",
        ":parse_utils",
        ":avro_utils",
        "@boost//:filesystem",
        ":concept",
        ":gem",
        ":rxnorm",
        ":umls",
    ]
)

cc_binary(
    name="extract_optum",
    srcs=[
        "extract_optum.cpp"
    ],
    deps = [
        ":lzma_utils",    
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        ":writer",
        ":parse_utils",
        "@readerwriterqueue",
    ]
)

cc_binary(
    name="clean_optum",
    srcs=[
        "clean_optum.cpp"
    ],
    deps = [    
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        ":reader",
        ":writer",
        ":umls",
        ":rxnorm",
        ":gem",
    ]
)

cc_binary(
    name="ontology_optum",
    srcs=[
        "ontology_optum.cpp"
    ],
    deps = [
        ":constdb",
        ":reader",
        ":umls",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],

)

cc_binary(
    name="create_index",
    srcs=[
        "create_index.cpp"
    ],
    deps = [
        ":constdb",
        ":reader",
        "@streamvbyte",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@boost//:filesystem",
    ]
)

cc_binary(
    name="featurize",
    srcs=[
        "featurize.cpp"
    ],
    deps = [
        ":constdb",
        ":reader",
        ":flatmap",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@boost//:filesystem",
        ":python",
        ":npy",
    ]
)


cc_library(
    name="npy",
    hdrs = [
        "npy.hpp"
    ],
)

cc_binary(
    name="extension.so",
    srcs = [
        "extension.cpp",
        "timeline_extension.cpp",
        "timeline_extension.h",
        "index_extension.cpp",
        "index_extension.h",
        "ontology_extension.cpp",
        "ontology_extension.h",
        "patient2vec_extension.cpp",
        "patient2vec_extension.h",
        "civil_day_caster.h"
    ],
    deps = [
        ":python",
        "@pybind11",
        ":reader",
        ":numpy",
        "@concurrentqueue",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@picosha2",
        ":flatmap",
    ],
    linkstatic=1, 
    linkshared=1
)

