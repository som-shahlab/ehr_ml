#include <iostream>

#include "boost/filesystem.hpp"
#include "constdb.h"
#include "flatmap.h"
#include "npy.hpp"
#include "reader.h"

int main() {
    boost::filesystem::path root(
        "/share/pi/nigam/ethanid/starr_omop_cdm5_latest_extract");

    boost::filesystem::path final_extract = root / "final" / "extract.db";

    boost::filesystem::path final_index = root / "final" / "index.db";

    ExtractReader extract(final_extract.c_str(), true);
    ExtractReaderIterator iterator = extract.iter();
    std::cout << "Starting to process" << std::endl;

    std::vector<uint16_t> data;
    std::vector<uint16_t> indices;
    std::vector<uint32_t> indptr;
    indptr.push_back(0);

    std::vector<uint8_t> result_labels;
    std::vector<uint32_t> patient_ids;
    std::vector<uint16_t> patient_day_indices;
    std::vector<uint16_t> years;
    std::vector<uint16_t> ages;
    std::vector<uint32_t> reverse_mapper;

    std::map<uint32_t, uint32_t> code_counts;

    FlatMap<uint32_t> total_counts;

    uint32_t next_index = 1;
    FlatMap<uint32_t> dictionary;
    reverse_mapper.push_back(0);

    uint32_t death_code =
        *extract.get_dictionary().map("Death Type/OMOP4822053");

    uint32_t visit_code = *extract.get_dictionary().map("Visit/IP");

    std::cout << "Got death code " << death_code << std::endl;

    for (uint32_t patient_id : extract.get_patient_ids()) {
        code_counts.clear();
        bool found = iterator.process_patient(
            patient_id, [&](absl::CivilDay birth_date, uint32_t age,
                            const std::vector<uint32_t>& observations,
                            const std::vector<ObservationWithValue>&
                                observations_with_values) {
                for (uint32_t obs : observations) {
                    code_counts[obs] += 1;
                }
            });

        for (const auto& item : code_counts) {
            (*total_counts.find_or_insert(item.first, 0))++;
        }

        if (!found) {
            std::cout << absl::Substitute("Could not find patient id $0",
                                          patient_id)
                      << std::endl;
            abort();
        }
    }

    uint32_t processed = 0;
    for (uint32_t patient_id : extract.get_patient_ids()) {
        processed += 1;

        // if (processed == 1000) {
        //     break;
        // }

        if (processed % 1000000 == 0) {
            std::cout << absl::Substitute("Processed $0 out of $1", processed,
                                          extract.get_patient_ids().size())
                      << std::endl;
        }

        int32_t first_age = -1;
        uint32_t count = 0;
        uint32_t death_age = 0;
        uint32_t last_age = 0;

        bool found = iterator.process_patient(
            patient_id, [&](absl::CivilDay birth_date, uint32_t age,
                            const std::vector<uint32_t>& observations,
                            const std::vector<ObservationWithValue>&
                                observations_with_values) {
                count++;

                if (count == 2) {
                    first_age = age;
                }
                for (uint32_t obs : observations) {
                    if (obs == death_code) {
                        death_age = age;
                    }
                }

                last_age = age;
            });

        if (first_age == -1) {
            continue;
        }

        if (!found) {
            std::cout << absl::Substitute("Could not find patient id $0",
                                          patient_id)
                      << std::endl;
            abort();
        }

        int32_t prev_age = 0;
        count = 0;
        code_counts.clear();
        found = iterator.process_patient(
            patient_id, [&](absl::CivilDay birth_date, uint32_t age,
                            const std::vector<uint32_t>& observations,
                            const std::vector<ObservationWithValue>&
                                observations_with_values) {
                bool has_visit = false;

                for (const auto& obs_with_val : observations_with_values) {
                    if (obs_with_val.code == visit_code) {
                        has_visit = true;
                    }
                }

                bool is_alive = (death_age == 0) || (age < death_age);
                bool is_positive = (death_age - age) < 180;
                bool has_enough_history = (last_age - age) >= 180;

                prev_age = age;

                if (has_visit && is_alive && (count >= 2) &&
                    (is_positive || has_enough_history)) {
                    result_labels.push_back(is_positive);
                    patient_ids.push_back(patient_id);
                    patient_day_indices.push_back(count - 1);
                    absl::CivilDay day = birth_date + age;
                    years.push_back(day.year());

                    data.push_back(age);
                    indices.push_back(0);

                    for (const auto& item : code_counts) {
                        data.push_back(item.second);

                        uint32_t* index =
                            dictionary.find_or_insert(item.first, next_index);

                        if (*index == next_index) {
                            next_index++;

                            if (reverse_mapper.size() != *index) {
                                std::cout << "What? " << reverse_mapper.size()
                                          << " " << *index << std::endl;
                                abort();
                            }

                            reverse_mapper.push_back(item.first);
                        }

                        indices.push_back(*index);
                    }
                    indptr.push_back(indices.size());
                }

                for (uint32_t obs : observations) {
                    if (*total_counts.find(obs) > 1000) {
                        code_counts[obs] += 1;
                    }
                }

                count++;
            });

        if (!found) {
            std::cout << absl::Substitute("Could not find patient id $0",
                                          patient_id)
                      << std::endl;
            abort();
        }
    }

    std::cout << "Saving" << std::endl;
    std::cout << "Num codes" << reverse_mapper.size() << std::endl;

    auto write_vec = [](std::string filename, const auto& vec) {
        unsigned long shape[] = {vec.size()};
        npy::SaveArrayAsNumpy(
            "/share/pi/nigam/ethanid/ehr_ml/r01/data/" + filename + ".npy",
            false, 1, shape, vec);
    };

    std::cout << "Number of labels " << result_labels.size() << std::endl;

    write_vec("data", data);
    write_vec("indices", indices);
    write_vec("indptr", indptr);
    write_vec("labels", result_labels);
    write_vec("patient_ids", patient_ids);
    write_vec("patient_day_indices", patient_day_indices);
    write_vec("years", years);
    write_vec("reverse_mapper", reverse_mapper);
}
