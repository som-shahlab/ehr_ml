#ifndef LZMA_UTILS_INCLUDED
#define LZMA_UTILS_INCLUDED

#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "absl/strings/string_view.h"
#include "lzma.h"

const int buffer_size = 1024 * 1024;  // One megabyte

template <typename F>
void stream_zx(const std::string& filename, F f) {
    lzma_stream strm = LZMA_STREAM_INIT;
    lzma_ret ret = lzma_stream_decoder(&strm, UINT64_MAX, LZMA_CONCATENATED);

    if (ret != LZMA_OK) {
        std::cout << "Could not initialize decoder " << ret << std::endl;
        exit(-1);
    }

    FILE* infile = fopen(filename.c_str(), "rb");
    if (infile == nullptr) {
        std::cout << "Could not open " << filename << " got error "
                  << strerror(errno) << std::endl;
        exit(-1);
    }

    std::vector<char> last_line;
    std::vector<char> inbuf(buffer_size);
    std::vector<char> outbuf(buffer_size);

    lzma_action action = LZMA_RUN;

    strm.next_in = NULL;
    strm.avail_in = 0;
    strm.next_out = (uint8_t*)outbuf.data();
    strm.avail_out = outbuf.size();

    while (true) {
        if (strm.avail_in == 0 && !feof(infile)) {
            strm.next_in = (uint8_t*)inbuf.data();
            strm.avail_in = fread(inbuf.data(), 1, sizeof(inbuf), infile);

            if (ferror(infile)) {
                fprintf(stderr, "%s: Read error: %s\n", filename.c_str(),
                        strerror(errno));
                exit(-1);
            }

            if (feof(infile)) {
                action = LZMA_FINISH;
            }
        }

        lzma_ret ret = lzma_code(&strm, action);

        if (strm.avail_out == 0 || ret == LZMA_STREAM_END) {
            size_t end_index = outbuf.size() - strm.avail_out;

            size_t last_newline = 0;

            for (size_t i = 0; i < end_index; i++) {
                char current_char = outbuf[i];

                if (current_char == '\n') {
                    // Need to purge
                    if (last_newline == 0) {
                        f(absl::string_view(last_line.data(),
                                            last_line.size()));
                        last_line.clear();
                    } else {
                        f(absl::string_view(outbuf.data() + last_newline,
                                            i - last_newline));
                    }
                    last_newline = i + 1;
                } else {
                    if (last_newline == 0) {
                        last_line.push_back(current_char);
                    }
                }
            }

            if (last_newline != 0) {
                last_line.insert(std::end(last_line),
                                 outbuf.data() + last_newline,
                                 outbuf.data() + end_index);
            }

            strm.next_out = (uint8_t*)outbuf.data();
            strm.avail_out = outbuf.size();
        }

        if (ret == LZMA_STREAM_END) {
            return;
        } else if (ret != LZMA_OK) {
            std::cout << "Got decompress error " << filename << " " << ret
                      << std::endl;
            exit(-1);
        }
    }
}

void split_into_fields(std::vector<absl::string_view>& fields,
                       absl::string_view text) {
    fields.clear();

    size_t last_comma = 0;
    bool in_quotes = false;
    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];

        if (in_quotes) {
            if (c == '"') {
                in_quotes = false;
            }
        } else {
            if (c == ',') {
                fields.push_back(
                    text.substr(last_comma + 1, (i - last_comma) - 2));
                last_comma = i + 1;
            } else if (c == '"') {
                in_quotes = true;
            }
        }
    }
    fields.push_back(text.substr(last_comma + 1, text.size() - 1));
}

template <typename F>
void parse_xz_csv(const std::string& filename,
                  const std::vector<std::string_view>& field_names, F f) {
    std::vector<absl::string_view> fields;
    std::vector<size_t> field_indices(field_names.size());
    bool parsed_header = false;

    std::vector<absl::string_view> results(field_names.size());

    stream_zx(filename, [&](absl::string_view text) {
        split_into_fields(fields, text);

        if (!parsed_header) {
            for (size_t j = 0; j < field_names.size(); j++) {
                bool found = false;
                for (size_t i = 0; i < fields.size(); i++) {
                    if (fields[i] == field_names[j]) {
                        found = true;
                        field_indices[j] = i;
                    }
                }

                if (!found) {
                    std::cout << "Could not find field name " << field_names[j]
                              << " for " << filename << std::endl;
                    std::cout << "Actually had: ";
                    for (size_t i = 0; i < fields.size(); i++) {
                        std::cout << "'" << fields[i] << "' ";
                    }
                    std::cout << std::endl;
                    exit(-1);
                }
            }
            parsed_header = true;
        } else {
            for (size_t j = 0; j < field_names.size(); j++) {
                results[j] = fields[field_indices[j]];
            }
            f(results);
        }
    });
}

#endif