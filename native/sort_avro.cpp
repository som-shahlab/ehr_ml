#include <boost/filesystem.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>

#include "avro.h"
#include "blockingconcurrentqueue.h"

void sort_file(boost::filesystem::path source,
               boost::filesystem::path destination) {
    // std::cout<<"Sorting " << source << " and putting into " << destination <<
    // std::endl;

    auto size = boost::filesystem::file_size(source);

    if (size < 1024 * 10) {  // 10 kilobytes
        std::cout << source << " is too small to work with " << std::endl;
        return;
    }

    avro_file_reader_t reader;

    int error = avro_file_reader(source.c_str(), &reader);
    if (error != 0) {
        std::cout << "Failed due to error opening " << source << " " << error
                  << " " << avro_strerror() << std::endl;
        abort();
    }

    avro_schema_t schema = avro_file_reader_get_writer_schema(reader);

    int person_id_index =
        avro_schema_record_field_get_index(schema, "person_id");

    if (person_id_index == -1) {
        std::cout << "Could not get person_id for " << source << std::endl;
        avro_file_reader_close(reader);
        avro_schema_decref(schema);
        return;
    }

    avro_file_writer_t writer;

    error = avro_file_writer_create_with_codec(destination.c_str(), schema,
                                               &writer, "snappy", 1024 * 1024);

    if (error != 0) {
        std::cout << "Failed due to error opening " << destination << std::endl;
        abort();
    }

    avro_value_iface_t* iface = avro_generic_class_from_schema(schema);

    std::vector<std::pair<int64_t, avro_value_t>> data_elements;

    int rval;

    while (true) {
        avro_value_t value;
        avro_generic_value_new(iface, &value);

        rval = avro_file_reader_read_value(reader, &value);

        if (rval == 0) {
            avro_value_t person_id_union_field;

            error = avro_value_get_by_index(&value, person_id_index,
                                            &person_id_union_field, nullptr);

            if (error != 0) {
                std::cout << "Could not get person_id field " << error
                          << std::endl;
                abort();
            }

            avro_value_t person_id_field;

            error = avro_value_get_current_branch(&person_id_union_field,
                                                  &person_id_field);
            if (error != 0) {
                std::cout << "Could not get person_id union field " << error
                          << std::endl;
                abort();
            }

            int64_t person_id;
            error = avro_value_get_long(&person_id_field, &person_id);

            if (error != 0) {
                std::cout << "Could not get person_id int64_t " << error
                          << std::endl;
                abort();
            }

            data_elements.push_back(std::make_pair(person_id, value));
        } else {
            avro_value_decref(&value);

            if (rval != EOF) {
                std::cout << "Could not read item " << rval << std::endl;
                abort();
            }

            break;
        }
    }

    std::sort(std::begin(data_elements), std::end(data_elements),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    for (auto& pid_and_value : data_elements) {
        auto& value = pid_and_value.second;

        if (avro_file_writer_append_value(writer, &value)) {
            std::cout << "Error writing to " << destination << " "
                      << avro_strerror() << std::endl;
            abort();
        }

        avro_value_decref(&value);
    }

    avro_file_reader_close(reader);
    avro_file_writer_close(writer);
    avro_value_iface_decref(iface);
    avro_schema_decref(schema);
}

using WorkItem = std::pair<boost::filesystem::path, boost::filesystem::path>;
using WorkQueue = moodycamel::BlockingConcurrentQueue<std::optional<WorkItem>>;

void worker_thread(std::shared_ptr<WorkQueue> work_queue) {
    while (true) {
        std::optional<WorkItem> result;
        work_queue->wait_dequeue(result);

        if (!result) {
            break;
        } else {
            auto& source = result->first;
            auto& target = result->second;

            sort_file(source, target);
        }
    }
}

int main() {
    boost::filesystem::path root(
        "/share/pi/nigam/ethanid/starr_omop_cdm5_latest_extract");

    boost::filesystem::path source_files = root / "source";
    boost::filesystem::path target_files = root / "sorted";

    std::vector<std::pair<boost::filesystem::path, boost::filesystem::path>>
        files_to_sort;

    std::shared_ptr<WorkQueue> work_queue = std::make_shared<WorkQueue>();

    for (auto&& file : boost::filesystem::directory_iterator(source_files)) {
        auto path = file.path();
        boost::filesystem::path target = target_files / path.filename();
        work_queue->enqueue(std::make_pair(path, target));
    }

    int num_threads = 10;

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; i++) {
        std::thread thread([work_queue]() { worker_thread(work_queue); });

        threads.push_back(std::move(thread));

        work_queue->enqueue(std::nullopt);
    }

    std::cout << "Joining" << std::endl;

    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "Done" << std::endl;

    // int a = avro_file_reader()
}