#include <dirent.h>
#include <glob.h>
#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "atomicops.h"
#include "avro.h"
#include "avro_utils.h"
#include "boost/filesystem.hpp"
#include "parse_utils.h"
#include "readerwriterqueue.h"
#include "writer.h"

struct RawPatientRecord {
    uint64_t person_id;
    std::optional<absl::CivilDay> birth_date;
    std::vector<std::pair<absl::CivilDay, uint32_t>> observations;
    std::vector<std::pair<absl::CivilDay, std::pair<uint32_t, uint32_t>>>
        observations_with_values;
    std::vector<uint32_t> birth_observations;
};

using QueueItem = std::variant<RawPatientRecord, Metadata>;

class Queue {
   public:
    Queue(size_t max_size)
        : inner_queue(max_size), capacity(max_size), count(0) {}

    void wait_enqueue(QueueItem&& item) {
        while (!capacity.wait())
            ;
        bool value = inner_queue.try_enqueue(item);
        if (!value) {
            std::cout << "Invariant failed in queue enqueue" << std::endl;
            abort();
        }
        count.signal();
    }

    void wait_dequeue(QueueItem& item) {
        while (!count.wait())
            ;
        bool value = inner_queue.try_dequeue(item);
        if (!value) {
            std::cout << "Invariant failed in queue dequeue" << std::endl;
            abort();
        }
        capacity.signal();
    }

   private:
    moodycamel::ReaderWriterQueue<QueueItem> inner_queue;
    moodycamel::spsc_sema::LightweightSemaphore capacity;
    moodycamel::spsc_sema::LightweightSemaphore count;
};

using WriterItem = std::variant<PatientRecord, Metadata>;

class Converter {
   public:
    virtual std::string_view get_file_prefix() const = 0;
    virtual std::vector<std::string_view> get_columns() const = 0;

    virtual void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                             const std::vector<avro_value_t>& row) const = 0;

    void run_converter(boost::filesystem::path path, Queue& queue) const {
        Metadata meta;

        size_t num_rows = 0;

        RawPatientRecord current_record;
        current_record.person_id = 0;

        std::vector<std::string_view> columns(get_columns());
        columns.push_back("person_id");

        parse_avro_file(path, columns, [&](const auto& row) {
            num_rows++;
            if (num_rows % 100000000 == 0) {
                std::cout << absl::Substitute("Processed $0 rows for $1\n",
                                              num_rows, path.string());
            }

            uint64_t person_id = (uint64_t)get_long(row[row.size() - 1]);

            if (person_id != current_record.person_id) {
                if (current_record.person_id) {
                    queue.wait_enqueue({std::move(current_record)});
                }

                current_record = {};
                current_record.person_id = person_id;
            }

            augment_day(meta, current_record, row);
        });

        if (current_record.person_id) {
            queue.wait_enqueue({std::move(current_record)});
        }

        std::cout << absl::Substitute("Done working on $0\n", path.string());

        queue.wait_enqueue({std::move(meta)});
    }

    virtual ~Converter() {}
};

class PatientConverter : public Converter {
   public:
    std::string_view get_file_prefix() const { return "person0"; }
    std::vector<std::string_view> get_columns() const {
        return {"birth_DATETIME", "gender_source_concept_id",
                "ethnicity_source_concept_id", "race_source_concept_id"};
    }

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<avro_value_t>& row) const {
        std::string_view birth_date = get_string(row[0]);

        absl::Time parsed_birth_date;

        std::string err;

        bool parsed = absl::ParseTime("%Y-%m-%d%ET%H:%M:%S", birth_date,
                                      &parsed_birth_date, &err);

        if (!parsed) {
            std::cout << "Got error parsing " << birth_date << " " << err
                      << std::endl;
        }

        patient_record.birth_date =
            absl::ToCivilDay(parsed_birth_date, absl::UTCTimeZone());

        uint32_t gender_code =
            meta.dictionary.map_or_add(get_long_string(row[1]));
        patient_record.birth_observations.push_back(gender_code);

        uint32_t ethnicity_code =
            meta.dictionary.map_or_add(get_long_string(row[2]));
        patient_record.birth_observations.push_back(ethnicity_code);

        uint32_t race_code =
            meta.dictionary.map_or_add(get_long_string(row[3]));
        patient_record.birth_observations.push_back(race_code);
    }
};

class StandardConceptTableConverter : public Converter {
   public:
    StandardConceptTableConverter(std::string f, std::string d, std::string c)
        : prefix(f), date_field(d), concept_id_field(c) {}

    std::string_view get_file_prefix() const override { return prefix; }

    std::vector<std::string_view> get_columns() const override {
        return {date_field, concept_id_field};
    }

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<avro_value_t>& row) const {
        patient_record.observations.push_back(std::make_pair(
            get_date(row[0]),
            meta.dictionary.map_or_add(get_long_string(row[1]))));
    }

   private:
    std::string prefix;
    std::string date_field;
    std::string concept_id_field;
};

class VisitConverter : public Converter {
 public:
    std::string_view get_file_prefix() const override { return "visit_occurrence0"; }

    std::vector<std::string_view> get_columns() const {
        return {"visit_start_DATE", "visit_concept_id", "visit_end_DATE"};
    }

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<avro_value_t>& row) const {
        std::string code = std::to_string(get_long(row[1]));

        if (code == "0") {
            return;
        }

        auto start_day = get_date(row[0]);
        auto end_day = get_date(row[2]);
        
        int days = end_day - start_day;

        ObservationWithValue obs;
        obs.code = meta.dictionary.map_or_add(code);
        obs.is_text = false;
        obs.numeric_value = days;

        patient_record.observations_with_values.push_back(
            std::make_pair(start_day, obs.encode()));
    }
};

class MeasurementConverter : public Converter {
   public:
    std::string_view get_file_prefix() const override { return "measurement0"; }

    std::vector<std::string_view> get_columns() const {
        return {"measurement_DATE", "measurement_concept_id", "value_as_number",
                "value_source_value"};
    }

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<avro_value_t>& row) const {
        std::string code = std::to_string(get_long(row[1]));
        std::string_view value;

        if (!is_null(row[3]) && get_string(row[3]) != "") {
            value = get_string(row[3]);
        } else if (!is_null(row[2])) {
            value = std::to_string(get_double(row[2]));
        } else {
            value = "";
        }

        auto day = get_date(row[0]);

        if (value == "") {
            patient_record.observations.push_back(
                std::make_pair(day, meta.dictionary.map_or_add(code)));
        } else {
            ObservationWithValue obs;
            obs.code = meta.dictionary.map_or_add(code);

            float numeric_value;
            bool is_valid_numeric = absl::SimpleAtof(value, &numeric_value);

            if (is_valid_numeric) {
                obs.is_text = false;
                obs.numeric_value = numeric_value;
            } else {
                obs.is_text = true;
                obs.text_value = meta.value_dictionary.map_or_add(value);
            }

            patient_record.observations_with_values.push_back(
                std::make_pair(day, obs.encode()));
        }
    }
};

std::pair<std::thread, std::shared_ptr<Queue>> generate_converter_thread(
    std::shared_ptr<Converter> converter, boost::filesystem::path path) {
    std::shared_ptr<Queue> queue =
        std::make_shared<Queue>(10000);  // Ten thousand patient records

    std::thread thread([converter, path, queue]() {
        std::string thread_name = path.filename().string();
        thread_name = thread_name.substr(0, 15);
        std::string name_copy(std::begin(thread_name), std::end(thread_name));
        int error = pthread_setname_np(pthread_self(), name_copy.c_str());
        if (error != 0) {
            std::cout << "Could not set thread name to " << thread_name << " "
                      << error << std::endl;
            abort();
        }
        converter->run_converter(path, *queue);
    });

    return std::make_pair(std::move(thread), std::move(queue));
}

class HeapItem {
   public:
    HeapItem(size_t _index, QueueItem _item)
        : index(_index), item(std::move(_item)) {}

    bool operator<(const HeapItem& second) const {
        std::optional<uint64_t> first_person_id = get_person_id();
        std::optional<uint64_t> second_person_id = second.get_person_id();

        uint64_t limit = std::numeric_limits<uint64_t>::max();

        return first_person_id.value_or(limit) >
               second_person_id.value_or(limit);
    }

    std::optional<uint64_t> get_person_id() const {
        return std::visit(
            [](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;

                if constexpr (std::is_same_v<T, RawPatientRecord>) {
                    return std::optional<uint64_t>(arg.person_id);
                } else {
                    return std::optional<uint64_t>();
                }
            },
            item);
    }

    size_t index;
    QueueItem item;
};

class Merger {
   public:
    Merger(std::vector<std::pair<std::thread, std::shared_ptr<Queue>>>
               _converter_threads)
        : converter_threads(std::move(_converter_threads)) {
        for (size_t i = 0; i < converter_threads.size(); i++) {
            const auto& entry = converter_threads[i];
            QueueItem nextItem;
            entry.second->wait_dequeue(nextItem);
            heap.push_back(HeapItem(i, std::move(nextItem)));
        }

        shift = 0;

        while ((((size_t)1) << shift) < converter_threads.size()) {
            shift++;
        }

        std::make_heap(std::begin(heap), std::end(heap));
    }

    WriterItem operator()() {
        while (true) {
            std::optional<uint64_t> possible_person_id =
                heap.front().get_person_id();

            if (possible_person_id.has_value()) {
                RawPatientRecord total_record;
                total_record.person_id = possible_person_id.value();
                total_record.birth_date =
                    absl::CivilDay(9999, 1, 1);  // Dummy value

                contributing_indexes.clear();

                while (heap.front().get_person_id() == possible_person_id) {
                    std::pop_heap(std::begin(heap), std::end(heap));
                    HeapItem& heap_item = heap.back();
                    QueueItem& queue_item = heap_item.item;

                    size_t index = heap_item.index;
                    RawPatientRecord& record =
                        std::get<RawPatientRecord>(queue_item);

                    if (record.birth_date &&
                        record.birth_date < total_record.birth_date) {
                        total_record.birth_date = record.birth_date;
                    }

                    auto offset = [&](uint32_t val) {
                        return (val << shift) + index;
                    };

                    for (const auto& obs : record.observations) {
                        total_record.observations.push_back(
                            std::make_pair(obs.first, offset(obs.second)));

                        if (obs.first < total_record.birth_date) {
                            total_record.birth_date = obs.first;
                        }
                    }

                    for (const auto& obs : record.birth_observations) {
                        total_record.birth_observations.push_back(offset(obs));
                    }

                    for (const auto& obs : record.observations_with_values) {
                        ObservationWithValue obs_with_value(obs.second.first,
                                                            obs.second.second);

                        obs_with_value.code = offset(obs_with_value.code);

                        if (obs_with_value.is_text) {
                            obs_with_value.text_value =
                                offset(obs_with_value.text_value);
                        }

                        total_record.observations_with_values.push_back(
                            std::make_pair(obs.first, obs_with_value.encode()));

                        if (obs.first < total_record.birth_date) {
                            total_record.birth_date = obs.first;
                        }
                    }

                    converter_threads[index].second->wait_dequeue(queue_item);
                    contributing_indexes.push_back(index);

                    std::push_heap(std::begin(heap), std::end(heap));
                }

                total_patients++;

                if (!total_record.birth_date) {
                    lost_patients++;

                    if (rand() % lost_patients == 0) {
                        std::cout
                            << "You have a patient without a birth date?? "
                            << total_record.person_id << " so far "
                            << lost_patients << " out of " << total_patients
                            << std::endl;
                        for (const auto& c_index : contributing_indexes) {
                            char thread_name[16];
                            pthread_getname_np(converter_threads[c_index]
                                                   .first.native_handle(),
                                               thread_name,
                                               sizeof(thread_name));
                            std::cout << "Thread: " << thread_name << std::endl;
                        }
                        std::cout << std::endl;
                    }

                    continue;
                }

                if (contributing_indexes.size() == 1) {
                    // Only got the member detail thread
                    ignored_patients++;

                    if (rand() % ignored_patients == 0) {
                        std::cout << "You are ignoring a patient "
                                  << total_record.person_id << " so far "
                                  << ignored_patients << " out of "
                                  << total_patients << std::endl;
                    }

                    continue;
                }

                PatientRecord final_record;
                final_record.person_id = total_record.person_id;

                final_record.birth_date = *total_record.birth_date;

                for (const auto& observ : total_record.observations) {
                    final_record.observations.push_back(std::make_pair(
                        observ.first - final_record.birth_date, observ.second));
                }

                for (const auto& observ : total_record.birth_observations) {
                    final_record.observations.push_back(
                        std::make_pair(0, observ));
                }

                for (const auto& observ :
                     total_record.observations_with_values) {
                    final_record.observations_with_values.push_back(
                        std::make_pair(observ.first - final_record.birth_date,
                                       observ.second));
                }

                return final_record;
            } else {
                Metadata total_metadata;

                std::vector<std::pair<std::string, uint32_t>> dictionary;
                std::vector<std::pair<std::string, uint32_t>> value_dictionary;

                for (auto& heap_item : heap) {
                    QueueItem& queue_item = heap_item.item;

                    size_t index = heap_item.index;
                    Metadata& meta = std::get<Metadata>(queue_item);

                    auto offset = [&](uint32_t val) {
                        return (val << shift) + index;
                    };

                    auto process =
                        [&](const TermDictionary& source,
                            std::vector<std::pair<std::string, uint32_t>>&
                                target) {
                            auto vals = source.decompose();
                            size_t target_size = vals.size() << shift;
                            target.resize(std::max(target_size, target.size()));

                            for (size_t i = 0; i < vals.size(); i++) {
                                target[offset(i)] = std::move(vals[i]);
                            }
                        };

                    process(meta.dictionary, dictionary);
                    process(meta.value_dictionary, value_dictionary);
                }

                total_metadata.dictionary =
                    TermDictionary(std::move(dictionary));
                total_metadata.value_dictionary =
                    TermDictionary(std::move(value_dictionary));

                std::cout << "Done with " << lost_patients
                          << " lost patients and " << ignored_patients
                          << " ignored patients out of " << total_patients
                          << std::endl;

                return total_metadata;
            }
        }
    }

    ~Merger() {
        std::cout << "Joining threads" << std::endl;

        for (auto& entry : converter_threads) {
            entry.first.join();
        }

        std::cout << "Done joining" << std::endl;
    }

   private:
    int lost_patients = 0;
    int ignored_patients = 0;
    int total_patients = 0;
    size_t shift;
    std::vector<size_t> contributing_indexes;
    std::vector<HeapItem> heap;
    std::vector<std::pair<std::thread, std::shared_ptr<Queue>>>
        converter_threads;
};

bool has_prefix(std::string_view a, std::string_view b) {
    return a.substr(0, b.size()) == b;
}

int main() {
    boost::filesystem::path root(
        "/share/pi/nigam/ethanid/starr_omop_cdm5_latest_extract");

    boost::filesystem::path target_files = root / "sorted";

    boost::filesystem::path raw_extract = root / "raw";

    std::vector<std::shared_ptr<Converter>> converters;
    std::vector<std::pair<std::thread, std::shared_ptr<Queue>>>
        converter_threads;

    converters.push_back(std::make_shared<PatientConverter>());

    converters.push_back(std::make_shared<StandardConceptTableConverter>(
        "drug_exposure", "drug_exposure_start_DATE", "drug_source_concept_id"));

    converters.push_back(std::make_shared<StandardConceptTableConverter>(
        "death", "death_DATE", "death_type_concept_id"));

    converters.push_back(std::make_shared<StandardConceptTableConverter>(
        "condition_occurrence0", "condition_start_DATE",
        "condition_source_concept_id"));

    converters.push_back(std::make_shared<StandardConceptTableConverter>(
        "procedure_occurrence", "procedure_DATE",
        "procedure_source_concept_id"));

    converters.push_back(std::make_shared<StandardConceptTableConverter>(
        "device_exposure", "device_exposure_start_DATE",
        "device_source_concept_id"));

    converters.push_back(std::make_shared<VisitConverter>());

    converters.push_back(std::make_shared<MeasurementConverter>());

    for (auto&& file : boost::filesystem::directory_iterator(target_files)) {
        auto path = file.path();

        std::string filename = path.filename().string();
        std::string_view filename_view(filename);

        for (const auto& converter : converters) {
            if (has_prefix(filename_view, converter->get_file_prefix())) {
                converter_threads.push_back(
                    generate_converter_thread(converter, path));
            }
        }
    }

    write_timeline(raw_extract.c_str(), Merger(std::move(converter_threads)));
}
