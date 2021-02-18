#include "extract_extension.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string_view>

namespace py = pybind11;

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "concept.h"
#include "constdb.h"
#include "csv.h"
#include "gem.h"
#include "reader.h"
#include "umls.h"
#include "writer.h"
#include "sort_csv.h"

std::vector<std::string> map_terminology_type(std::string_view terminology) {
    if (terminology == "CPT4") {
        return {"CPT"};
    } else if (terminology == "LOINC") {
        return {"LNC"};
    } else if (terminology == "HCPCS") {
        return {"HCPCS", "CDT"};
    } else {
        return {std::string(terminology)};
    }
}

std::optional<std::string> try_to_recover(const UMLS& umls,
                                          const ConceptTable& table,
                                          uint32_t concept_id) {
    ConceptInfo info = *table.get_info(concept_id);

    std::vector<uint32_t> new_id_candidates;

    for (const auto& relationship : table.get_relationships(concept_id)) {
        if (relationship.relationship_id == "Is a") {
            new_id_candidates.push_back(relationship.other_concept);
        }
    }

    if (info.vocabulary_id == "ICD10CM") {
        // Hack to work around weird ICD10 behavior ...
        std::vector<std::pair<uint32_t, uint32_t>> ids_with_lengths;
        for (uint32_t candidate : new_id_candidates) {
            ConceptInfo c_info = *table.get_info(candidate);
            ids_with_lengths.push_back(
                std::make_pair(-c_info.concept_code.size(), candidate));
        }
        std::sort(std::begin(ids_with_lengths), std::end(ids_with_lengths));

        new_id_candidates.clear();

        for (auto pair : ids_with_lengths) {
            new_id_candidates.push_back(pair.second);
        }

        if (new_id_candidates.size() > 1) {
            new_id_candidates.resize(1);
        }
    }

    if (new_id_candidates.size() == 0) {
        // Could not find a replacement
        return std::nullopt;
    } else if (new_id_candidates.size() > 1) {
        std::cout << "Odd " << info.vocabulary_id << " " << info.concept_code
                  << " " << concept_id << " " << new_id_candidates.size()
                  << std::endl;
        return std::nullopt;
    } else {
        const auto& info = *table.get_info(new_id_candidates[0]);

        for (std::string terminology :
             map_terminology_type(info.vocabulary_id)) {
            auto res = umls.get_aui(terminology, info.concept_code);

            if (res) {
                return *res;
            }
        }

        return try_to_recover(umls, table, new_id_candidates[0]);
    }
}

std::vector<uint32_t> compute_subwords(
    std::string aui, const UMLS& umls,
    absl::flat_hash_map<std::string, std::vector<uint32_t>>&
        aui_to_subwords_map,
    absl::flat_hash_map<uint32_t, std::vector<uint32_t>>& code_to_parents_map,
    TermDictionary& dictionary) {
    if (aui_to_subwords_map.find(aui) == std::end(aui_to_subwords_map)) {
        std::vector<uint32_t> results;
        auto parents = umls.get_parents(aui);

        auto info = umls.get_code(aui);

        if (!info) {
            std::cout << "Could not find " << aui << std::endl;
            abort();
        }

        std::string word = absl::Substitute("$0/$1", info->first, info->second);

        uint32_t aui_code = dictionary.map_or_add(word);

        results.push_back(aui_code);

        for (const auto& parent_aui : parents) {
            auto parent_info_iter = umls.get_code(parent_aui);

            if (!parent_info_iter) {
                std::cout << "Could not find " << parent_aui << std::endl;
                abort();
            }

            std::string parent_word = absl::Substitute(
                "$0/$1", parent_info_iter->first, parent_info_iter->second);

            code_to_parents_map[aui_code].push_back(
                dictionary.map_or_add(parent_word));

            for (uint32_t subword :
                 compute_subwords(parent_aui, umls, aui_to_subwords_map,
                                  code_to_parents_map, dictionary)) {
                results.push_back(subword);
            }
        }

        if (std::find(std::begin(results), std::end(results),
                      dictionary.map_or_add("SRC/V-SRC")) ==
            std::end(results)) {
            std::cout << "AUI " << aui << " has no root " << std::endl;

            for (const auto& item : results) {
                std::cout << "Got " << dictionary.get_word(item).value()
                          << std::endl;
            }

            return {};
        }

        std::sort(std::begin(results), std::end(results));
        results.erase(std::unique(std::begin(results), std::end(results)),
                      std::end(results));
        aui_to_subwords_map.insert(std::make_pair(aui, std::move(results)));
    }

    return aui_to_subwords_map.find(aui)->second;
}

void create_ontology(std::string_view root_path, std::string umls_path,
                     std::string cdm_location, const ConceptTable& table) {
    std::string source = absl::Substitute("$0/extract.db", root_path);
    ExtractReader extract(source.c_str(), false);

    const TermDictionary& dictionary = extract.get_dictionary();

    auto entries = dictionary.decompose();

    UMLS umls(umls_path);

    TermDictionary ontology_dictionary;

    std::string ontology_path = absl::Substitute("$0/ontology.db", root_path);
    ConstdbWriter ontology_writer(ontology_path.c_str());

    absl::flat_hash_map<std::string, std::vector<uint32_t>> aui_to_subwords_map;
    absl::flat_hash_map<uint32_t, std::vector<uint32_t>> code_to_parents_map;

    std::vector<uint32_t> words_with_subwords;
    std::vector<uint32_t> recorded_date_codes;

    std::set<std::string> code_types;

    std::set<std::string> recorded_date_code_types = {
        "ATC",   "CPT4",    "DRG",      "Gender",
        "HCPCS", "ICD10CM", "ICD10PCS", "LOINC"};

    for (uint32_t i = 0; i < entries.size(); i++) {
        const auto& word = entries[i].first;
        std::vector<uint32_t> subwords = {};

        std::vector<std::string_view> parts = absl::StrSplit(word, '/');

        if (parts.size() != 2) {
            std::cout << "Got weird vocab string " << word << std::endl;
            abort();
        }

        code_types.insert(std::string(parts[0]));

        if (recorded_date_code_types.find(std::string(parts[0])) !=
            std::end(recorded_date_code_types)) {
            recorded_date_codes.push_back(i);
        }

        std::optional<std::string> result = std::nullopt;

        for (std::string terminology : map_terminology_type(parts[0])) {
            auto res = umls.get_aui(terminology, std::string(parts[1]));
            if (res) {
                result = res;
            }
        }

        if (result == std::nullopt &&
            (parts[0] == "CPT4" || parts[0] == "ICD10CM")) {
            // Manually try to recover by using the OMOP hierarchy to map to
            // something useful.

            std::optional<uint32_t> opt_concept_id =
                table.get_inverse(std::string(parts[0]), std::string(parts[1]));

            if (!opt_concept_id) {
                std::cout << "Could not get inverse concept id " << word
                          << std::endl;
                abort();
            }

            uint32_t concept_id = *opt_concept_id;
            result = try_to_recover(umls, table, concept_id);
        }

        if (result == std::nullopt) {
            subwords = {ontology_dictionary.map_or_add(
                absl::Substitute("NO_MAP/$0", word))};
        } else {
            subwords =
                compute_subwords(*result, umls, aui_to_subwords_map,
                                 code_to_parents_map, ontology_dictionary);
        }

        ontology_writer.add_int(i, (const char*)subwords.data(),
                                subwords.size() * sizeof(uint32_t));
        words_with_subwords.push_back(i);
    }

    for (auto& iter : code_to_parents_map) {
        auto& parent_codes = iter.second;
        std::sort(std::begin(parent_codes), std::end(parent_codes));

        int32_t subword_as_int = iter.first + 1;
        ontology_writer.add_int(-subword_as_int,
                                (const char*)parent_codes.data(),
                                parent_codes.size() * sizeof(uint32_t));
    }

    for (auto& type : code_types) {
        std::cout << "Got type " << type << std::endl;
    }

    std::string dictionary_str = ontology_dictionary.to_json();
    ontology_writer.add_str("dictionary", dictionary_str.data(),
                            dictionary_str.size());
    ontology_writer.add_str("words_with_subwords",
                            (const char*)words_with_subwords.data(),
                            words_with_subwords.size() * sizeof(uint32_t));
    ontology_writer.add_str("recorded_date_codes",
                            (const char*)recorded_date_codes.data(),
                            recorded_date_codes.size() * sizeof(uint32_t));
    uint32_t root_node = *ontology_dictionary.map("SRC/V-SRC");
    ontology_writer.add_str("root", (const char*)&root_node, sizeof(uint32_t));
}

void create_index(std::string_view root_path) {
    std::string read = absl::Substitute("$0/extract.db", root_path);
    ExtractReader extract(read.c_str(), true);
    ExtractReaderIterator iterator = extract.iter();

    absl::flat_hash_map<uint32_t, std::vector<uint32_t>> patients_per_code;

    std::cout << "Starting to process" << std::endl;

    std::vector<uint32_t> codes;
    uint32_t processed = 0;
    for (uint32_t patient_id : extract.get_patient_ids()) {
        processed += 1;

        if (processed % 1000000 == 0) {
            std::cout << absl::Substitute("Processed $0 out of $1", processed,
                                          extract.get_patient_ids().size())
                      << std::endl;
        }
        codes.clear();
        bool found = iterator.process_patient(
            patient_id, [&codes](absl::CivilDay birth_date, uint32_t age,
                                 const std::vector<uint32_t>& observations,
                                 const std::vector<ObservationWithValue>&
                                     observations_with_values) {
                for (uint32_t obs : observations) {
                    codes.push_back(obs);
                }

                for (auto obs_with_value : observations_with_values) {
                    codes.push_back(obs_with_value.code);
                }
            });

        if (!found) {
            std::cout << absl::Substitute("Could not find patient id $0",
                                          patient_id)
                      << std::endl;
            abort();
        }

        std::sort(std::begin(codes), std::end(codes));
        codes.erase(std::unique(std::begin(codes), std::end(codes)),
                    std::end(codes));

        for (uint32_t code : codes) {
            patients_per_code[code].push_back(patient_id);
        }
    }

    std::string target = absl::Substitute("$0/index.db", root_path);
    ConstdbWriter writer(target.c_str());

    std::vector<uint8_t> compressed_buffer;
    for (auto& item : patients_per_code) {
        uint32_t code_id = item.first;
        std::vector<uint32_t>& patient_ids = item.second;

        std::sort(std::begin(patient_ids), std::end(patient_ids));

        uint32_t last_id = 0;

        for (uint32_t& pid : patient_ids) {
            uint32_t delta = pid - last_id;
            last_id = pid;
            pid = delta;
        }

        size_t max_needed_size =
            streamvbyte_max_compressedbytes(patient_ids.size()) +
            sizeof(uint32_t);

        if (compressed_buffer.size() < max_needed_size) {
            compressed_buffer.resize(max_needed_size * 2 + 1);
        }

        size_t actual_size =
            streamvbyte_encode(patient_ids.data(), patient_ids.size(),
                               compressed_buffer.data() + sizeof(uint32_t));

        uint32_t* start_of_compressed_buffer =
            reinterpret_cast<uint32_t*>(compressed_buffer.data());
        *start_of_compressed_buffer = patient_ids.size();

        writer.add_int(code_id, (const char*)compressed_buffer.data(),
                       actual_size + sizeof(uint32_t));
    }
}

struct RawPatientRecord {
    uint64_t person_id;
    std::optional<absl::CivilDay> birth_date;
    std::vector<std::pair<absl::CivilDay, uint32_t>> observations;
    std::vector<std::pair<absl::CivilDay, std::pair<uint32_t, uint32_t>>>
        observations_with_values;
};

using QueueItem = std::variant<RawPatientRecord, Metadata>;
using Queue = BlockingQueue<QueueItem>;

absl::CivilDay parse_date(std::string_view datestr) {
    std::string_view time_column = datestr;
    auto location = time_column.find(' ');
    if (location != std::string_view::npos) {
        time_column = time_column.substr(0, location);
    }

    location = time_column.find('T');
    if (location != std::string_view::npos) {
        time_column = time_column.substr(0, location);
    }

    auto first_dash = time_column.find('-');
    int year;
    attempt_parse_or_die(time_column.substr(0, first_dash), year);
    time_column = time_column.substr(first_dash + 1, std::string_view::npos);

    auto second_dash = time_column.find('-');
    int month;
    attempt_parse_or_die(time_column.substr(0, second_dash), month);
    time_column = time_column.substr(second_dash + 1, std::string_view::npos);

    int day;
    attempt_parse_or_die(time_column, day);

    return absl::CivilDay(year, month, day);
}

class Converter {
   public:
    std::string_view get_file_prefix() const;
    std::vector<std::string_view> get_columns() const;

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<std::string_view>& row) const;
};

template <typename C>
void run_converter(C converter, Queue& queue, boost::filesystem::path file) {
    Metadata meta;

    size_t num_rows = 0;

    RawPatientRecord current_record;
    current_record.person_id = 0;

    std::vector<std::string_view> columns = converter.get_columns();
    columns.push_back("person_id");

    csv_iterator(
        file.c_str(), columns, ',', {}, true, false, [&](const auto& row) {
            num_rows++;

            if (num_rows % 100000000 == 0) {
                std::cout << absl::Substitute("Processed $0 rows for $1\n",
                                              num_rows, file.string());
            }

            uint64_t person_id;
            attempt_parse_or_die(row[columns.size() - 1], person_id);

            if (person_id != current_record.person_id) {
                if (current_record.person_id) {
                    queue.wait_enqueue({std::move(current_record)});
                }

                current_record = {};
                current_record.person_id = person_id;
            }

            converter.augment_day(meta, current_record, row);
        });

    if (current_record.person_id) {
        queue.wait_enqueue({std::move(current_record)});
    }

    std::cout << absl::Substitute("Done working on $0\n", file.string());

    queue.wait_enqueue({std::move(meta)});
}

class DemographicsConverter : public Converter {
   public:
    std::string_view get_file_prefix() const { return "person"; }

    std::vector<std::string_view> get_columns() const {
        return {"birth_DATETIME", "gender_source_concept_id",
                "ethnicity_source_concept_id", "race_source_concept_id"};
    }

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<std::string_view>& row) const {
        absl::CivilDay birth = parse_date(row[0]);
        patient_record.birth_date = birth;

        uint32_t gender_code = meta.dictionary.map_or_add(row[1]);
        patient_record.observations.push_back(
            std::make_pair(birth, gender_code));

        uint32_t ethnicity_code = meta.dictionary.map_or_add(row[2]);
        patient_record.observations.push_back(
            std::make_pair(birth, ethnicity_code));

        uint32_t race_code = meta.dictionary.map_or_add(row[3]);
        patient_record.observations.push_back(std::make_pair(birth, race_code));
    }
};

class StandardConceptTableConverter : public Converter {
   public:
    StandardConceptTableConverter(std::string f, std::string d, std::string c)
        : prefix(f), date_field(d), concept_id_field(c) {}

    std::string_view get_file_prefix() const { return prefix; }

    std::vector<std::string_view> get_columns() const {
        return {date_field, concept_id_field};
    }

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<std::string_view>& row) const {
        patient_record.observations.push_back(std::make_pair(
            parse_date(row[0]), meta.dictionary.map_or_add(row[1])));
    }

   private:
    std::string prefix;
    std::string date_field;
    std::string concept_id_field;
};

class VisitConverter : public Converter {
   public:
    std::string_view get_file_prefix() const { return "visit_occurrence"; }

    std::vector<std::string_view> get_columns() const {
        return {"visit_start_DATE", "visit_concept_id", "visit_end_DATE"};
    }

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<std::string_view>& row) const {
        std::string_view code = row[1];

        if (code == "0") {
            return;
        }

        auto start_day = parse_date(row[0]);
        auto end_day = parse_date(row[2]);

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
    std::string_view get_file_prefix() const { return "measurement"; }

    std::vector<std::string_view> get_columns() const {
        return {"measurement_DATE", "measurement_concept_id", "value_as_number",
                "value_source_value"};
    }

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<std::string_view>& row) const {
        std::string_view code = row[1];
        std::string_view value;

        if (row[3] != "") {
            value = row[3];
        } else if (row[2] != "") {
            value = row[2];
        } else {
            value = "";
        }

        auto day = parse_date(row[0]);

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

template <typename C>
std::pair<std::thread, std::shared_ptr<Queue>> generate_converter_thread(
    const C& converter, boost::filesystem::path target) {
    std::shared_ptr<Queue> queue =
        std::make_shared<Queue>(10000);  // Ten thousand patient records
    std::thread thread([converter, queue, target]() {
        std::string thread_name = target.string();
        thread_name = thread_name.substr(0, 15);
        std::string name_copy(std::begin(thread_name), std::end(thread_name));
        int error = pthread_setname_np(pthread_self(), name_copy.c_str());
        if (error != 0) {
            std::cout << "Could not set thread name to " << thread_name << " "
                      << error << std::endl;
            abort();
        }
        run_converter(std::move(converter), *queue, target);
    });

    return std::make_pair(std::move(thread), std::move(queue));
}

template <typename C>
std::vector<std::pair<std::thread, std::shared_ptr<Queue>>>
generate_converter_threads(
    const C& converter,
    const std::vector<boost::filesystem::path>& possible_targets) {
    std::vector<std::pair<std::thread, std::shared_ptr<Queue>>> results;

    std::vector<std::string> options = {
        absl::Substitute("/$0/", converter.get_file_prefix()),
        absl::Substitute("/$0.csv.gz", converter.get_file_prefix()),
        absl::Substitute("/$00", converter.get_file_prefix()),
    };

    for (const auto& target : possible_targets) {
        bool found = false;
        for (const auto& option : options) {
            if (target.string().find(option) != std::string::npos &&
                target.string().find(".csv.gz") != std::string::npos) {
                found = true;
                break;
            }
        }
        if (found) {
            results.push_back(generate_converter_thread(converter, target));
        }
    }

    return results;
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

        if (heap.size() == 0) {
            std::cout << "No converters in the heap?" << std::endl;
            abort();
        }
    }

    WriterItem operator()() {
        while (true) {
            std::optional<uint64_t> possible_person_id =
                heap.front().get_person_id();

            if (possible_person_id.has_value()) {
                RawPatientRecord total_record;
                total_record.person_id = possible_person_id.value();
                total_record.birth_date = {};

                contributing_indexes.clear();

                while (heap.front().get_person_id() == possible_person_id) {
                    std::pop_heap(std::begin(heap), std::end(heap));
                    HeapItem& heap_item = heap.back();
                    QueueItem& queue_item = heap_item.item;

                    size_t index = heap_item.index;
                    RawPatientRecord& record =
                        std::get<RawPatientRecord>(queue_item);

                    if (record.birth_date) {
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
                    // Only got the person thread
                    ignored_patients++;

                    if (rand() % ignored_patients == 0) {
                        std::cout << "You are ignoring a patient "
                                  << total_record.person_id << " so far "
                                  << ignored_patients << " out of "
                                  << total_patients << std::endl;
                    }

                    continue;
                } else {
                    if (rand() % total_patients == 0) {
                        std::cout << "You finished a patient "
                                  << total_record.person_id << " out of "
                                  << total_patients << std::endl;
                    }
                }

                PatientRecord final_record;
                final_record.person_id = total_record.person_id;

                final_record.birth_date = *total_record.birth_date;

                for (const auto& observ : total_record.observations) {
                    final_record.observations.push_back(std::make_pair(
                        observ.first - final_record.birth_date, observ.second));
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

std::function<std::optional<PatientRecord>()> convert_vector_to_iter(
    std::vector<PatientRecord> r) {
    size_t index = 0;
    return [index, records = std::move(r)]() mutable {
        if (index == records.size()) {
            return std::optional<PatientRecord>();
        } else {
            return std::optional<PatientRecord>(std::move(records[index++]));
        }
    };
}

TermDictionary counts_to_dict(
    const absl::flat_hash_map<std::string, uint32_t>& counts) {
    TermDictionary result;

    std::vector<std::pair<uint32_t, std::string>> entries;

    for (const auto& iter : counts) {
        entries.push_back(std::make_pair(iter.second, iter.first));
    }

    std::sort(std::begin(entries), std::end(entries),
              std::greater<std::pair<uint32_t, std::string>>());

    for (const auto& row : entries) {
        result.map_or_add(row.second, row.first);
    }

    return result;
}

std::vector<std::string> normalize(
    std::string input_code, const ConceptTable& table, const GEMMapper& gem,
    absl::flat_hash_map<uint32_t, std::vector<uint32_t>>& rxnorm_to_atc) {
    if (input_code == "" || input_code == "0") {
        return {};
    }

    uint32_t concept_id;
    attempt_parse_or_die(input_code, concept_id);

    std::set<std::string> good_items = {"LOINC",
                                        "ICD10CM",
                                        "CPT4",
                                        "Gender",
                                        "HCPCS",
                                        "Ethnicity",
                                        "Race",
                                        "ICD10PCS",
                                        "Condition Type",
                                        "Visit",
                                        "CMS Place of Service"};
    std::set<std::string> bad_items = {"SNOMED",       "NDC",
                                       "ICD10CN",      "ICD10",
                                       "ICD9ProcCN",   "STANFORD_CONDITION",
                                       "STANFORD_MEAS"};

    std::vector<uint32_t> results;

    ConceptInfo info = *table.get_info(concept_id);
    if (info.vocabulary_id == "RxNorm") {
        // Need to map NDC over to ATC to avoid painful issues

        uint32_t rxnorm_code;
        attempt_parse_or_die(info.concept_code, rxnorm_code);

        auto iter = rxnorm_to_atc.find(rxnorm_code);
        if (iter == std::end(rxnorm_to_atc)) {
            std::vector<uint32_t> atc_codes;
            for (const auto& ancestor : table.get_ancestors(rxnorm_code)) {
                const auto& anc_info = *table.get_info(ancestor);
                if (anc_info.vocabulary_id == "RxNorm" ||
                    anc_info.vocabulary_id == "RxNorm Extension") {
                    for (const auto& relationship :
                         table.get_relationships(ancestor)) {
                        ConceptInfo other_info =
                            *table.get_info(relationship.other_concept);
                        if (other_info.vocabulary_id == "ATC") {
                            atc_codes.push_back(relationship.other_concept);
                        }
                    }
                }
            }

            std::sort(std::begin(atc_codes), std::end(atc_codes));
            atc_codes.erase(
                std::unique(std::begin(atc_codes), std::end(atc_codes)),
                std::end(atc_codes));

            // if (atc_codes.size() == 0) {
            //     std::cout << absl::Substitute("Could not find any atc
            //     codes for $0\n", rxnorm_code);
            // }

            rxnorm_to_atc[rxnorm_code] = atc_codes;
            iter = rxnorm_to_atc.find(rxnorm_code);
        }

        results = iter->second;
    } else if (info.vocabulary_id == "ICD9Proc") {
        for (const auto& proc : gem.map_proc(info.concept_code)) {
            auto new_code = table.get_inverse("ICD10PCS", proc);
            if (!new_code) {
                std::cout << absl::Substitute(
                    "Could not find $0 after converting $1\n", proc,
                    info.concept_code);
            }
            results.push_back(*new_code);
        }
    } else if (info.vocabulary_id == "ICD9CM") {
        for (std::string diag : gem.map_diag(info.concept_code)) {
            auto new_code = table.get_inverse("ICD10CM", diag);
            if (!new_code) {
                std::cout << absl::Substitute(
                    "Could not find $0 after converting $1\n", diag,
                    info.concept_code);
            }
            results.push_back(*new_code);
        }
    } else if (good_items.find(info.vocabulary_id) != std::end(good_items)) {
        results.push_back(concept_id);
    } else if (bad_items.find(info.vocabulary_id) != std::end(bad_items)) {
        return {};
    } else {
        std::cout << "Could not handle '" << info.vocabulary_id << "' '"
                  << input_code << "'" << std::endl;
        return {};
    }

    std::vector<std::string> final_results;

    for (const auto result : results) {
        ConceptInfo result_info = *table.get_info(result);

        if (result_info.vocabulary_id == "Condition Type") {
            std::string final =
                absl::Substitute("$0/$1", result_info.concept_class_id,
                                 result_info.concept_code);

            final_results.push_back(final);
        } else {
            std::string final = absl::Substitute(
                "$0/$1", result_info.vocabulary_id, result_info.concept_code);

            final_results.push_back(final);
        }
    }

    return final_results;
}

class Cleaner {
   public:
    Cleaner(const ConceptTable& concepts, const GEMMapper& gem,
            const char* path)
        : reader(path, false), iterator(reader.iter()) {
        patient_ids = reader.get_patient_ids();
        original_patient_ids = reader.get_original_patient_ids();
        current_index = 0;

        {
            TermDictionary temp_dictionary;
            std::vector<std::pair<std::string, uint32_t>> items =
                reader.get_dictionary().decompose();

            remap_dict.reserve(items.size());

            absl::flat_hash_map<std::string, uint32_t> lost_counts;

            absl::flat_hash_map<uint32_t, std::vector<uint32_t>> rxnorm_to_atc;

            uint32_t total_lost = 0;

            for (const auto& entry : items) {
                std::vector<std::string> terms =
                    normalize(entry.first, concepts, gem, rxnorm_to_atc);
                if (terms.size() == 0) {
                    total_lost += entry.second;
                    lost_counts[entry.first] += entry.second;
                }

                std::vector<uint32_t> result;
                for (const auto& term : terms) {
                    result.push_back(
                        temp_dictionary.map_or_add(term, entry.second));
                }

                remap_dict.push_back(result);
            }

            std::cout << "Lost items " << total_lost << std::endl;

            std::vector<std::pair<int32_t, std::string>> lost_entries;

            for (const auto& entry : lost_counts) {
                lost_entries.push_back(
                    std::make_pair(-entry.second, entry.first));
                // std::cout<<entry.first << " " << entry.second << std::endl;
            }
            std::sort(std::begin(lost_entries), std::end(lost_entries));

            for (size_t i = 0; i < 30 && i < lost_entries.size(); i++) {
                const auto& entry = lost_entries[i];
                std::cout << entry.second << " " << entry.first << std::endl;
            }

            auto [a, b] = temp_dictionary.optimize();
            final_dictionary = a;

            for (auto& entry : remap_dict) {
                for (auto& val : entry) {
                    val = b[val];
                }
            }
        }

        {
            TermDictionary temp_dictionary;

            std::vector<std::pair<std::string, uint32_t>> items =
                reader.get_value_dictionary().decompose();
            value_remap_dict.reserve(items.size());

            for (const auto& entry : items) {
                value_remap_dict.push_back(
                    temp_dictionary.map_or_add(entry.first, entry.second));
            }

            auto [a, b] = temp_dictionary.optimize();
            final_value_dictionary = a;

            for (auto& entry : value_remap_dict) {
                entry = b[entry];
            }
        }

        std::cout << "Dictionary size " << reader.get_dictionary().size()
                  << std::endl;
        std::cout << "Value dictionary size "
                  << reader.get_value_dictionary().size() << std::endl;
        std::cout << "Final Dictionary size " << final_dictionary.size()
                  << std::endl;
        std::cout << "Final Value dictionary size "
                  << final_value_dictionary.size() << std::endl;
        std::cout << "Num patients " << patient_ids.size() << std::endl;
    }

    WriterItem operator()() {
        if (current_index == patient_ids.size()) {
            Metadata meta;
            meta.dictionary = final_dictionary;
            meta.value_dictionary = final_value_dictionary;
            return meta;
        } else {
            uint32_t patient_id = patient_ids[current_index];

            PatientRecord record;
            record.person_id = original_patient_ids[current_index];
            current_index++;

            iterator.process_patient(
                patient_id, [&](absl::CivilDay birth_date, uint32_t age,
                                const std::vector<uint32_t>& observations,
                                const std::vector<ObservationWithValue>&
                                    observations_with_values) {
                    record.birth_date = birth_date;

                    for (const auto& obs : observations) {
                        for (const auto& remapped : remap_dict[obs]) {
                            record.observations.push_back(
                                std::make_pair(age, remapped));
                        }
                    }

                    for (const auto& obs_with_value :
                         observations_with_values) {
                        for (const auto& remapped_code :
                             remap_dict[obs_with_value.code]) {
                            ObservationWithValue new_obs;
                            new_obs.code = remapped_code;

                            if (obs_with_value.is_text) {
                                new_obs.is_text = true;
                                new_obs.text_value =
                                    value_remap_dict[obs_with_value.text_value];
                            } else {
                                new_obs.is_text = false;
                                new_obs.numeric_value =
                                    obs_with_value.numeric_value;
                            }

                            record.observations_with_values.push_back(
                                std::make_pair(age, new_obs.encode()));
                        }
                    }
                });

            return record;
        }
    }

   private:
    ExtractReader reader;

    std::vector<std::vector<uint32_t>> remap_dict;
    std::vector<uint32_t> value_remap_dict;

    TermDictionary final_dictionary;
    TermDictionary final_value_dictionary;

    ExtractReaderIterator iterator;
    absl::Span<const uint32_t> patient_ids;
    absl::Span<const uint64_t> original_patient_ids;
    size_t current_index;
};

void create_extract(std::string omop_source_dir, std::string target_directory, const ConceptTable& concepts, const GEMMapper& gem) {
    std::vector<std::pair<std::thread, std::shared_ptr<Queue>>>
        converter_threads;

    std::vector<boost::filesystem::path> targets;

    for (const auto& p :
         boost::filesystem::recursive_directory_iterator(omop_source_dir)) {
        targets.push_back(p);
    }

    auto helper = [&](const auto& c) {
        auto results = generate_converter_threads(c, targets);

        for (auto& result : results) {
            converter_threads.push_back(std::move(result));
        }
    };

    helper(DemographicsConverter());
    helper(VisitConverter());
    helper(MeasurementConverter());
    helper(StandardConceptTableConverter(
        "drug_exposure", "drug_exposure_start_date", "drug_source_concept_id"));
    helper(StandardConceptTableConverter("death", "death_date",
                                         "death_type_concept_id"));
    helper(StandardConceptTableConverter("condition_occurrence",
                                         "condition_start_date",
                                         "condition_source_concept_id"));
    helper(StandardConceptTableConverter("procedure_occurrence",
                                         "procedure_date",
                                         "procedure_source_concept_id"));
    helper(StandardConceptTableConverter("device_exposure",
                                         "device_exposure_start_date",
                                         "device_source_concept_id"));
    helper(StandardConceptTableConverter("observation", "observation_date",
                                         "observation_source_concept_id"));

    std::string tmp_extract = absl::Substitute("$0/tmp.db", target_directory);
    std::string final_extract =
        absl::Substitute("$0/extract.db", target_directory);

    write_timeline(tmp_extract.c_str(), Merger(std::move(converter_threads)));

    write_timeline(final_extract.c_str(),
                   Cleaner(concepts, gem, tmp_extract.c_str()));

    
    boost::filesystem::remove(tmp_extract);
}

void perform_omop_extraction(std::string omop_source_dir_str, std::string umls_dir,
                             std::string gem_dir,
                             std::string target_dir_str) {

    boost::filesystem::path omop_source_dir = boost::filesystem::canonical(omop_source_dir_str);
    boost::filesystem::path target_dir = boost::filesystem::weakly_canonical(target_dir_str);

    if (!boost::filesystem::create_directory(target_dir)) {
        std::cout << absl::Substitute(
            "Could not make result directory $0, got error $1\n",
            target_dir.string(), std::strerror(errno));
        exit(-1);
    }

    boost::filesystem::path sorted_dir = target_dir / "sorted";

    boost::filesystem::create_directory(sorted_dir);

    sort_csvs(omop_source_dir, sorted_dir);

    ConceptTable concepts = construct_concept_table(omop_source_dir.string());
    GEMMapper gem(gem_dir);

    create_extract(sorted_dir.string(), target_dir.string(), concepts, gem);

    boost::filesystem::remove_all(sorted_dir); 

    create_index(target_dir.string());
    create_ontology(target_dir.string(), umls_dir, omop_source_dir.string(), concepts);
}

void register_extract_extension(py::module& root) {
    py::module m = root.def_submodule("extract");

    m.def("extract_omop", perform_omop_extraction);
}