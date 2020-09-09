#ifndef AVRO_UTILS_H_INCLUDED
#define AVRO_UTILS_H_INCLUDED

#include <iostream>

#include "avro.h"
#include "boost/filesystem.hpp"

bool is_null(const avro_value_t& value) {
    return (avro_value_get_null(&value) == 0);
}

std::string_view get_string(const avro_value_t& value) {
    const char* pointer;
    size_t size;
    int error = avro_value_get_string(&value, &pointer, &size);

    if (error != 0) {
        std::cout << "Could not get string " << std::endl;
        abort();
    }

    return std::string_view(pointer, size - 1);
}

absl::CivilDay get_date(const avro_value_t& value) {
    absl::string_view date_str = get_string(value);

    absl::Time parsed_date;

    std::string err;

    bool parsed = absl::ParseTime("%Y-%m-%d", date_str, &parsed_date, &err);

    if (!parsed) {
        std::cout << "Got error parsing " << date_str << " " << err
                  << std::endl;
    }

    return absl::ToCivilDay(parsed_date, absl::UTCTimeZone());
}

int64_t get_long(const avro_value_t& value) {
    int64_t result;
    int error = avro_value_get_long(&value, &result);

    if (error != 0) {
        std::cout << "Could not get long " << std::endl;
        abort();
    }

    return result;
}

double get_double(const avro_value_t& value) {
    double result;
    int error = avro_value_get_double(&value, &result);

    if (error != 0) {
        std::cout << "Could not get double " << std::endl;
        abort();
    }

    return result;
}

std::string get_long_string(const avro_value_t& value) {
    int64_t val = get_long(value);
    return absl::StrCat(val);
}

template <typename F>
void parse_avro_file(const boost::filesystem::path& path,
                     const std::vector<std::string_view>& columns, const F& f) {
    avro_file_reader_t reader;

    int error = avro_file_reader(path.c_str(), &reader);
    if (error != 0) {
        std::cout << "Failed due to error opening " << path << " " << error
                  << " " << avro_strerror() << std::endl;
        abort();
    }

    avro_schema_t schema = avro_file_reader_get_writer_schema(reader);

    std::vector<int> indices;

    for (const auto& column : columns) {
        std::string copy(column);
        int index = avro_schema_record_field_get_index(schema, copy.c_str());

        if (index == -1) {
            std::cout << "Could not find column " << column << " in " << path
                      << std::endl;
            abort();
        }

        indices.push_back(index);
    }

    avro_value_iface_t* iface = avro_generic_class_from_schema(schema);

    avro_value_t value;
    avro_generic_value_new(iface, &value);

    int rval;

    std::vector<avro_value_t> row(indices.size());

    while (true) {
        rval = avro_file_reader_read_value(reader, &value);

        if (rval == 0) {
            for (size_t i = 0; i < indices.size(); i++) {
                int index = indices[i];

                avro_value_t union_field;

                error = avro_value_get_by_index(&value, index, &union_field,
                                                nullptr);

                if (error != 0) {
                    std::cout << "Could not get field " << error << std::endl;
                    abort();
                }

                avro_value_t& target = row[i];

                error = avro_value_get_current_branch(&union_field, &target);
                if (error != 0) {
                    std::cout << "Could not get union field " << error
                              << std::endl;
                    abort();
                }
            }

            f(row);

            avro_value_reset(&value);
        } else {
            if (rval != EOF) {
                std::cout << "Could not read item " << rval << std::endl;
                abort();
            }
            break;
        }
    }

    avro_value_decref(&value);
    avro_file_reader_close(reader);
    avro_value_iface_decref(iface);
    avro_schema_decref(schema);
}

#endif