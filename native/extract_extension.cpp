#include "extract_extension.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string_view>

namespace py = pybind11;

#include <dirent.h>
#include <sys/stat.h>

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

        std::vector<absl::string_view> parts = absl::StrSplit(word, '/');

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

struct PatientInfo {
    PatientInfo(uint64_t p_id, uint32_t i, absl::CivilDay b)
        : person_id(p_id), index(i), birth_date(b) {}

    uint64_t person_id;
    uint32_t index;
    absl::CivilDay birth_date;
};

absl::flat_hash_map<uint64_t, PatientInfo> get_patient_data(
    std::string location) {
    std::string person_file =
        absl::Substitute("$0/$1", location, "person.csv.gz");

    std::vector<std::string_view> columns = {"person_id", "year_of_birth"};

    absl::flat_hash_map<uint64_t, PatientInfo> result;

    int i = 0;

    csv_iterator(person_file.c_str(), columns, '\t', {}, true,
                 [&i, &result](const auto& row) {
                     int64_t person_id;
                     int year_of_birth;
                     attempt_parse_or_die(row[0], person_id);
                     attempt_parse_or_die(row[1], year_of_birth);

                     absl::CivilDay birth_date(year_of_birth, 1, 1);

                     result.insert(std::make_pair(
                         person_id, PatientInfo(person_id, i++, birth_date)));
                 });

    return result;
}

class PatientRecord {
   public:
    uint32_t index;
    uint64_t person_id;
    absl::CivilDay birth_date;
    std::vector<std::pair<uint32_t, uint32_t>> observations;
    std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>>
        observationsWithValues;
};

absl::CivilDay parse_date(std::string_view datestr) {
    std::string_view time_column = datestr;
    auto location = time_column.find('T');
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
    std::string_view get_file() const;
    std::vector<std::string_view> get_columns() const;

    absl::CivilDay get_date(absl::CivilDay birth_date,
                            const std::vector<std::string_view>& row) const {
        return parse_date(row[1]);
    }

    void augment_day(TermDictionary& dictionary,
                     TermDictionary& value_dictionary,
                     PatientRecord& patient_record, uint32_t day_index,
                     const std::vector<std::string_view>& row) const;
};

class DemographicsConverter : public Converter {
   public:
    std::string_view get_file() const { return "person.csv.gz"; }

    std::vector<std::string_view> get_columns() const {
        return {"person_id", "gender_concept_id", "race_concept_id",
                "ethnicity_concept_id"};
    }

    absl::CivilDay get_date(absl::CivilDay birth_date,
                            const std::vector<std::string_view>& row) const {
        return birth_date;
    }

    void augment_day(TermDictionary& dictionary,
                     TermDictionary& value_dictionary,
                     PatientRecord& patient_record, uint32_t day_index,
                     const std::vector<std::string_view>& row) const {
        for (int i = 1; i < 4; i++) {
            if (row[i] != "" && row[i] != "0") {
                patient_record.observations.push_back(
                    std::make_pair(day_index, dictionary.map_or_add(row[i])));
            }
        }
    }
};

class StandardConceptTableConverter : public Converter {
   public:
    StandardConceptTableConverter(std::string f, std::string d, std::string c)
        : filename(f), date_field(d), concept_id_field(c) {}

    std::string_view get_file() const { return filename; }

    std::vector<std::string_view> get_columns() const {
        return {"person_id", date_field, concept_id_field};
    }

    void augment_day(TermDictionary& dictionary,
                     TermDictionary& value_dictionary,
                     PatientRecord& patient_record, uint32_t day_index,
                     const std::vector<std::string_view>& row) const {
        patient_record.observations.push_back(
            std::make_pair(day_index, dictionary.map_or_add(row[2])));
    }

   private:
    std::string filename;
    std::string date_field;
    std::string concept_id_field;
};

class VisitConverter : public Converter {
   public:
    std::string_view get_file() const { return "visit_occurrence.csv.gz"; }

    std::vector<std::string_view> get_columns() const {
        return {"person_id", "visit_start_date", "visit_end_date",
                "visit_concept_id"};
    }

    void augment_day(TermDictionary& dictionary,
                     TermDictionary& value_dictionary,
                     PatientRecord& patient_record, uint32_t day_index,
                     const std::vector<std::string_view>& row) const {
        uint32_t duration;

        if (row[2] != "") {
            auto start_date = parse_date(row[1]);
            auto end_date = parse_date(row[2]);
            duration = end_date - start_date;
        } else {
            duration = 0;
        }

        ObservationWithValue obs;
        obs.code = dictionary.map_or_add(row[3]);
        obs.is_text = false;
        obs.numeric_value = duration;

        patient_record.observationsWithValues.push_back(
            std::make_pair(day_index, obs.encode()));
    }
};

class MeasurementConverter : public Converter {
   public:
    std::string_view get_file() const { return "measurement.csv.gz"; }

    std::vector<std::string_view> get_columns() const {
        return {"person_id", "measurement_date",
                "measurement_source_concept_id", "value_as_number",
                "value_source_value"};
    }

    void augment_day(TermDictionary& dictionary,
                     TermDictionary& value_dictionary,
                     PatientRecord& patient_record, uint32_t day_index,
                     const std::vector<std::string_view>& row) const {
        std::string_view code = row[2];
        std::string_view value;

        if (row[4] != "") {
            value = row[4];
        } else {
            value = row[3];
        }

        if (value == "") {
            patient_record.observations.push_back(
                std::make_pair(day_index, dictionary.map_or_add(code)));
        } else {
            ObservationWithValue obs;
            obs.code = dictionary.map_or_add(code);

            float numeric_value;
            bool is_valid_numeric = absl::SimpleAtof(value, &numeric_value);

            if (is_valid_numeric) {
                obs.is_text = false;
                obs.numeric_value = numeric_value;
            } else {
                obs.is_text = true;
                obs.text_value = value_dictionary.map_or_add(value);
            }

            patient_record.observationsWithValues.push_back(
                std::make_pair(day_index, obs.encode()));
        }
    }
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

template <typename F>
void write_timeline(const char* filename, const TermDictionary& dictionary,
                    const TermDictionary& value_dictionary, F get_next) {
    std::cout << absl::Substitute("Writing to $0\n", filename);
    ConstdbWriter writer(filename);

    std::vector<uint64_t> original_ids;
    std::vector<uint32_t> patient_ids;

    std::vector<uint32_t> buffer;
    std::vector<uint8_t> compressed_buffer;
    std::vector<uint32_t> ages;

    while (true) {
        std::optional<PatientRecord> next_record = get_next();

        if (!next_record) {
            break;
        }

        PatientRecord& record = *next_record;

        buffer.clear();
        compressed_buffer.clear();
        ages.clear();

        if (record.person_id == 0) {
            continue;
        }

        patient_ids.push_back(record.index);
        original_ids.push_back(record.person_id);

        buffer.push_back(record.birth_date.year());
        buffer.push_back(record.birth_date.month());
        buffer.push_back(record.birth_date.day());

        std::sort(std::begin(record.observations),
                  std::end(record.observations));
        std::sort(std::begin(record.observationsWithValues),
                  std::end(record.observationsWithValues));

        record.observations.erase(std::unique(std::begin(record.observations),
                                              std::end(record.observations)),
                                  std::end(record.observations));
        record.observationsWithValues.erase(
            std::unique(std::begin(record.observationsWithValues),
                        std::end(record.observationsWithValues)),
            std::end(record.observationsWithValues));

        for (const auto& elem : record.observations) {
            ages.push_back(elem.first);
        }
        for (const auto& elem : record.observationsWithValues) {
            ages.push_back(elem.first);
        }

        std::sort(std::begin(ages), std::end(ages));
        ages.erase(std::unique(std::begin(ages), std::end(ages)),
                   std::end(ages));

        buffer.push_back(ages.size());

        uint32_t last_age = 0;

        size_t current_observation_index = 0;
        size_t current_observation_with_values_index = 0;

        for (uint32_t age : ages) {
            uint32_t delta = age - last_age;
            last_age = age;

            buffer.push_back(delta);

            size_t num_obs_index = buffer.size();
            buffer.push_back(1 << 30);  // Use a high value to force crashes

            size_t starting_observation_index = current_observation_index;
            uint32_t last_observation = 0;

            while (current_observation_index < record.observations.size() &&
                   record.observations[current_observation_index].first ==
                       age) {
                uint32_t current_obs =
                    record.observations[current_observation_index].second;
                uint32_t delta = current_obs - last_observation;
                last_observation = current_obs;
                buffer.push_back(delta);
                current_observation_index++;
            }

            buffer[num_obs_index] =
                current_observation_index - starting_observation_index;

            size_t num_obs_with_value_index = buffer.size();
            buffer.push_back(1 << 30);  // Use a high value to force crashes

            size_t starting_observation_value_index =
                current_observation_with_values_index;
            uint32_t last_observation_with_value = 0;

            while (current_observation_with_values_index <
                       record.observationsWithValues.size() &&
                   record.observationsWithValues
                           [current_observation_with_values_index]
                               .first == age) {
                auto [code, value] =
                    record
                        .observationsWithValues
                            [current_observation_with_values_index]
                        .second;
                uint32_t delta = code - last_observation_with_value;
                last_observation_with_value = code;
                buffer.push_back(delta);
                buffer.push_back(value);
                current_observation_with_values_index++;
            }

            buffer[num_obs_with_value_index] =
                current_observation_with_values_index -
                starting_observation_value_index;
        }

        size_t max_needed_size =
            streamvbyte_max_compressedbytes(buffer.size()) + sizeof(uint32_t);

        if (compressed_buffer.size() < max_needed_size) {
            compressed_buffer.resize(max_needed_size * 2 + 1);
        }

        size_t actual_size =
            streamvbyte_encode(buffer.data(), buffer.size(),
                               compressed_buffer.data() + sizeof(uint32_t));

        uint32_t* start_of_compressed_buffer =
            reinterpret_cast<uint32_t*>(compressed_buffer.data());
        *start_of_compressed_buffer = buffer.size();

        writer.add_int(record.index, (const char*)compressed_buffer.data(),
                       actual_size + sizeof(uint32_t));
    }

    uint32_t num_patients = original_ids.size();

    writer.add_str("num_patients", (const char*)&num_patients,
                   sizeof(uint32_t));
    writer.add_str("original_ids", (const char*)original_ids.data(),
                   sizeof(uint64_t) * original_ids.size());
    writer.add_str("patient_ids", (const char*)patient_ids.data(),
                   sizeof(uint32_t) * patient_ids.size());

    std::string dictionary_str = dictionary.to_json();
    std::string value_dictionary_str = value_dictionary.to_json();

    writer.add_str("dictionary", dictionary_str.data(), dictionary_str.size());
    writer.add_str("value_dictionary", value_dictionary_str.data(),
                   value_dictionary_str.size());

    std::cout << absl::Substitute("Done writing to $0\n", filename);
}

template <typename C>
std::thread generate_converter_thread(
    std::string location, std::string root_file, std::string name, C convert,
    const absl::flat_hash_map<uint64_t, PatientInfo>& patient_data) {
    return std::thread([location, convert, root_file, name, &patient_data]() {
        std::string file_name = absl::Substitute("$0/$1", root_file, name);
        run_converter(location, file_name, convert, patient_data);
    });
}

template <typename C>
void run_converter(
    std::string location, std::string_view filename, const C& convert,
    const absl::flat_hash_map<uint64_t, PatientInfo>& patient_data) {
    std::cout << absl::Substitute("Starting to work on $0\n", filename);
    int iter = 0;
    size_t num_rows = 0;

    TermDictionary dictionary;
    TermDictionary value_dictionary;

    std::string person_file =
        absl::Substitute("$0/$1", location, convert.get_file());

    std::vector<PatientRecord> results;

    csv_iterator(
        person_file.c_str(), convert.get_columns(), '\t', {}, true,
        [&](const auto& row) {
            num_rows++;

            if (num_rows % 100000000 == 0) {
                std::cout << absl::Substitute("Processed $0 rows\n", num_rows);
            }

            if (num_rows % 1000000000 == 0) {
                std::string current_filename =
                    absl::Substitute("$0_$1", filename, iter);

                write_timeline(current_filename.c_str(), dictionary,
                               value_dictionary,
                               convert_vector_to_iter(std::move(results)));

                dictionary.clear();
                value_dictionary.clear();
                results.clear();

                iter += 1;
            }

            uint64_t person_id;
            attempt_parse_or_die(row[0], person_id);

            auto iter = patient_data.find(person_id);
            const PatientInfo& info = iter->second;

            if (info.index >= results.size()) {
                results.resize(info.index * 2 + 1);
            }

            PatientRecord& record = results[info.index];
            record.birth_date = info.birth_date;
            record.person_id = info.person_id;
            record.index = info.index;

            auto date = convert.get_date(info.birth_date, row);

            int day_index = date - info.birth_date;

            if (day_index < 0) {
                return;
            }

            convert.augment_day(dictionary, value_dictionary, record, day_index,
                                row);
        });

    std::string current_filename = absl::Substitute("$0_$1", filename, iter);
    write_timeline(current_filename.c_str(), dictionary, value_dictionary,
                   convert_vector_to_iter(std::move(results)));
    std::cout << absl::Substitute("Done working on $0\n", filename);
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

std::pair<std::vector<std::vector<uint32_t>>, TermDictionary>
convert_to_concept_strings(const TermDictionary& terms,
                           const ConceptTable& table, const GEMMapper& gem) {
    auto entries = terms.decompose();
    std::vector<std::vector<uint32_t>> converter(entries.size());

    TermDictionary new_term_dictionary;
    absl::flat_hash_map<uint32_t, std::vector<uint32_t>> rxnorm_to_atc;

    for (uint32_t code = 0; code < entries.size(); code++) {
        auto [term, count] = entries[code];

        uint32_t concept_id;
        if (!absl::SimpleAtoi(term, &concept_id)) {
            std::cout << absl::Substitute(
                "Could not parse supposed concept_id $0\n", term);
        }

        if (concept_id == 0) {
            continue;
        }

        std::vector<uint32_t> results;

        std::optional<ConceptInfo> maybe_info = table.get_info(concept_id);
        if (!maybe_info) {
            continue;
        }
        ConceptInfo info = *maybe_info;
        if (info.vocabulary_id == "NDC") {
            // Need to map NDC over to ATC to avoid painful issues
            std::vector<uint32_t> rxnorm_codes;
            for (const auto& relationship :
                 table.get_relationships(concept_id)) {
                if (relationship.relationship_id == "Maps to") {
                    std::optional<ConceptInfo> other_info =
                        table.get_info(relationship.other_concept);
                    if (!other_info) {
                        continue;
                    }
                    if (other_info->vocabulary_id == "RxNorm" ||
                        other_info->vocabulary_id == "RxNorm Extension") {
                        rxnorm_codes.push_back(relationship.other_concept);
                    }
                }
            }

            if (rxnorm_codes.size() > 1) {
                std::cout << absl::Substitute(
                    "Got a weird number of RxNorm mappings for $0 with $1\n",
                    concept_id, rxnorm_codes.size());
                continue;
            }

            if (rxnorm_codes.size() == 0) {
                // std::cout << absl::Substitute("Could not find any rxnorm code
                // for $0\n", concept_id);
                continue;
            }

            uint32_t rxnorm_code = rxnorm_codes[0];

            auto iter = rxnorm_to_atc.find(rxnorm_code);
            if (iter == std::end(rxnorm_to_atc)) {
                std::vector<uint32_t> atc_codes;
                for (const auto& ancestor : table.get_ancestors(rxnorm_code)) {
                    const auto& anc_info = table.get_info(ancestor);
                    if (anc_info &&
                        (anc_info->vocabulary_id == "RxNorm" ||
                         anc_info->vocabulary_id == "RxNorm Extension")) {
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
        } else {
            results.push_back(concept_id);
        }

        for (const auto result : results) {
            std::optional<ConceptInfo> result_info = table.get_info(result);
            if (!result_info) {
                continue;
            }
            std::string final = absl::Substitute(
                "$0/$1", result_info->vocabulary_id, result_info->concept_code);
            converter[code].push_back(
                new_term_dictionary.map_or_add(final, count));
        }
    }

    return {converter, new_term_dictionary};
}

void merge_intermediates(std::string source_folder, std::string gem_location,
                         const std::vector<std::string>& files,
                         const std::string& target,
                         const ConceptTable& concepts) {
    GEMMapper gem(gem_location);

    std::cout << absl::Substitute("Processing the concept table\n");

    std::cout << absl::Substitute("Starting to merge results\n");
    std::vector<ExtractReader> readers;
    std::vector<std::vector<std::vector<uint32_t>>> all_converters;
    std::vector<std::vector<std::string>> all_dict_mappers;
    absl::flat_hash_map<std::string, uint32_t> dict_counts;

    std::vector<std::vector<std::string>> all_val_mappers;
    absl::flat_hash_map<std::string, uint32_t> val_counts;

    for (size_t i = 0; i < files.size(); i++) {
        const std::string& file = files[i];
        std::cout << absl::Substitute("Starting to load $0\n", file);
        readers.emplace_back(file.c_str(), true);

        ExtractReader& reader = readers[i];
        auto [converter, new_dict] =
            convert_to_concept_strings(reader.get_dictionary(), concepts, gem);
        auto dict_entries = new_dict.decompose();
        auto value_entries = reader.get_value_dictionary().decompose();

        std::cout << absl::Substitute(
            "$0 normal terms and $1 value terms for $2\n", dict_entries.size(),
            value_entries.size(), files[i]);

        std::vector<std::string> dict_mapper;
        for (const auto& entry : dict_entries) {
            dict_mapper.push_back(entry.first);
            dict_counts[entry.first] += entry.second;
        }

        std::vector<std::string> val_dict_mapper;
        for (const auto& entry : value_entries) {
            val_dict_mapper.push_back(entry.first);
            val_counts[entry.first] += entry.second;
        }

        all_converters.push_back(std::move(converter));
        all_dict_mappers.push_back(std::move(dict_mapper));
        all_val_mappers.push_back(std::move(val_dict_mapper));
    }

    std::vector<ExtractReaderIterator> iterators;

    for (size_t i = 0; i < files.size(); i++) {
        iterators.push_back(readers[i].iter());
    }

    std::cout << absl::Substitute(
        "Got a total of $0 normal terms and $1 value terms\n",
        dict_counts.size(), val_counts.size());

    TermDictionary final_dict = counts_to_dict(dict_counts);
    TermDictionary final_val_dict = counts_to_dict(val_counts);

    std::vector<std::vector<uint32_t>> remapper;
    std::vector<std::vector<uint32_t>> value_remapper;

    for (const auto& dict_mapper : all_dict_mappers) {
        std::vector<uint32_t> result(dict_mapper.size());
        for (size_t i = 0; i < dict_mapper.size(); i++) {
            result[i] = *final_dict.map(dict_mapper[i]);
        }
        remapper.push_back(std::move(result));
    }

    for (const auto& val_mapper : all_val_mappers) {
        std::vector<uint32_t> result(val_mapper.size());
        for (size_t i = 0; i < val_mapper.size(); i++) {
            result[i] = *final_val_dict.map(val_mapper[i]);
        }
        value_remapper.push_back(std::move(result));
    }

    std::vector<uint32_t> patient_record_indices(readers.size());

    auto iter = [&]() {
        uint32_t next_patient_id = std::numeric_limits<uint32_t>::max();
        uint64_t next_patient_original_id =
            std::numeric_limits<uint64_t>::max();

        for (size_t i = 0; i < patient_record_indices.size(); i++) {
            uint32_t index = patient_record_indices[i];
            if (index < readers[i].get_patient_ids().size() &&
                readers[i].get_patient_ids()[index] < next_patient_id) {
                next_patient_id = readers[i].get_patient_ids()[index];
                next_patient_original_id =
                    readers[i].get_original_patient_ids()[index];
            }
        }

        if (next_patient_id == std::numeric_limits<uint32_t>::max()) {
            return std::optional<PatientRecord>();
        }

        PatientRecord record;
        record.index = next_patient_id;

        for (size_t i = 0; i < patient_record_indices.size(); i++) {
            const auto& converter = all_converters.at(i);
            const auto& remap = remapper.at(i);
            const auto& value_remap = value_remapper.at(i);
            uint32_t& index = patient_record_indices[i];
            if (index < readers[i].get_patient_ids().size() &&
                readers[i].get_patient_ids()[index] == next_patient_id) {
                index++;

                bool found = iterators[i].process_patient(
                    next_patient_id,
                    [&](absl::CivilDay birth_date, uint32_t age,
                        const std::vector<uint32_t>& observations,
                        const std::vector<ObservationWithValue>&
                            observations_with_values) {
                        record.birth_date = birth_date;
                        record.person_id = next_patient_original_id;

                        for (uint32_t obs : observations) {
                            for (uint32_t converted : converter[obs]) {
                                record.observations.push_back(
                                    std::make_pair(age, remap[converted]));
                            }
                        }

                        for (auto obs_with_value : observations_with_values) {
                            for (uint32_t converted :
                                 converter[obs_with_value.code]) {
                                obs_with_value.code = remap[converted];

                                if (obs_with_value.is_text) {
                                    obs_with_value.text_value =
                                        value_remap[obs_with_value.text_value];
                                }

                                record.observationsWithValues.push_back(
                                    std::make_pair(age,
                                                   obs_with_value.encode()));
                            }
                        }
                    });

                if (!found) {
                    std::cout << absl::Substitute(
                        "Could not find patient_id $0\n", next_patient_id);
                    exit(-1);
                }
            }
        }

        return std::optional<PatientRecord>(std::move(record));
    };

    write_timeline(target.c_str(), final_dict, final_val_dict, iter);
}

void perform_omop_extraction(std::string omop_source_dir, std::string umls_dir,
                             std::string gem_dir,
                             std::string target_directory) {
    int error = mkdir(target_directory.c_str(), 0700);

    if (error == -1) {
        std::cout << absl::Substitute(
            "Could not make result directory $0, got error $1\n",
            target_directory, std::strerror(errno));
        exit(-1);
    }

    std::string root_path = absl::Substitute("$0/temp", target_directory);

    error = mkdir(root_path.c_str(), 0700);

    if (error == -1) {
        std::cout << absl::Substitute(
            "Could not make result directory $0, got error $1\n", root_path,
            std::strerror(errno));
        exit(-1);
    }

    std::cout << absl::Substitute("Starting to download patient data\n");
    auto patient_data = get_patient_data(omop_source_dir);
    std::cout << absl::Substitute("Found $0 patients\n", patient_data.size());

    std::vector<std::thread> converter_threads;
    converter_threads.push_back(
        generate_converter_thread(omop_source_dir, root_path, "demo",
                                  DemographicsConverter(), patient_data));
    converter_threads.push_back(generate_converter_thread(
        omop_source_dir, root_path, "visit", VisitConverter(), patient_data));
    converter_threads.push_back(
        generate_converter_thread(omop_source_dir, root_path, "measure",
                                  MeasurementConverter(), patient_data));
    converter_threads.push_back(generate_converter_thread(
        omop_source_dir, root_path, "drug",
        StandardConceptTableConverter("drug_exposure.csv.gz",
                                      "drug_exposure_start_date",
                                      "drug_source_concept_id"),
        patient_data));
    converter_threads.push_back(generate_converter_thread(
        omop_source_dir, root_path, "death",
        StandardConceptTableConverter("death.csv.gz", "death_date",
                                      "death_type_concept_id"),
        patient_data));
    converter_threads.push_back(generate_converter_thread(
        omop_source_dir, root_path, "cond",
        StandardConceptTableConverter("condition_occurrence.csv.gz",
                                      "condition_start_date",
                                      "condition_source_concept_id"),
        patient_data));
    converter_threads.push_back(generate_converter_thread(
        omop_source_dir, root_path, "proc",
        StandardConceptTableConverter("procedure_occurrence.csv.gz",
                                      "procedure_date",
                                      "procedure_source_concept_id"),
        patient_data));
    converter_threads.push_back(generate_converter_thread(
        omop_source_dir, root_path, "device",
        StandardConceptTableConverter("device_exposure.csv.gz",
                                      "device_exposure_start_date",
                                      "device_source_concept_id"),
        patient_data));
    converter_threads.push_back(generate_converter_thread(
        omop_source_dir, root_path, "obs",
        StandardConceptTableConverter("observation.csv.gz", "observation_date",
                                      "observation_source_concept_id"),
        patient_data));

    for (std::thread& thread : converter_threads) {
        thread.join();
    }

    std::cout << absl::Substitute("Finding files to merge\n");
    std::vector<std::string> files;

    DIR* d = opendir(root_path.c_str());
    if (d) {
        dirent* dir = readdir(d);
        while (dir != nullptr) {
            std::string filename = dir->d_name;
            dir = readdir(d);
            if (filename[0] == '.') {
                continue;
            }

            std::cout << absl::Substitute("Found $0\n", filename);
            files.push_back(absl::Substitute("$0/$1", root_path, filename));
        }
        closedir(d);
    }

    std::string target = absl::Substitute("$0/extract.db", target_directory);

    ConceptTable table = construct_concept_table_csv(omop_source_dir);

    merge_intermediates(omop_source_dir, gem_dir, files, target, table);
    create_index(target_directory);
    create_ontology(target_directory, umls_dir, omop_source_dir, table);
}

void register_extract_extension(py::module& root) {
    py::module m = root.def_submodule("extract");

    m.def("extract_omop", perform_omop_extraction);
}