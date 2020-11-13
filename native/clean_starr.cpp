#include <iostream>

#include "boost/filesystem.hpp"
#include "concept.h"
#include "gem.h"
#include "reader.h"
#include "rxnorm.h"
#include "umls.h"
#include "writer.h"

std::vector<std::string> normalize(
    std::string input_code, const ConceptTable& table, const GEMMapper& gem,
    absl::flat_hash_map<uint32_t, std::vector<uint32_t>>& rxnorm_to_atc) {
    if (input_code == "" || input_code == "0") {
        return {};
    }

    uint32_t concept_id;
    attempt_parse_or_die(input_code, concept_id);

    std::set<std::string> good_items = {"LOINC",  "ICD10CM", "CPT4",
                                        "Gender", "HCPCS",   "Ethnicity",
                                        "Race",   "ICD10PCS", "Condition Type", "Visit", "CMS Place of Service"};
    std::set<std::string> bad_items = {"SNOMED", "NDC",   "ICD10CN",
                                       "ICD10", "ICD9ProcCN"};

    std::vector<uint32_t> results;

    ConceptInfo info = table.get_info(concept_id);
    if (info.vocabulary_id == "RxNorm") {
        // Need to map NDC over to ATC to avoid painful issues

        uint32_t rxnorm_code;
        attempt_parse_or_die(info.concept_code, rxnorm_code);

        auto iter = rxnorm_to_atc.find(rxnorm_code);
        if (iter == std::end(rxnorm_to_atc)) {
            std::vector<uint32_t> atc_codes;
            for (const auto& ancestor : table.get_ancestors(rxnorm_code)) {
                const auto& anc_info = table.get_info(ancestor);
                if (anc_info.vocabulary_id == "RxNorm" ||
                    anc_info.vocabulary_id == "RxNorm Extension") {
                    for (const auto& relationship :
                         table.get_relationships(ancestor)) {
                        ConceptInfo other_info =
                            table.get_info(relationship.other_concept);
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
        std::cout << "Could not handle '" << info.vocabulary_id << "' '" << input_code << "'" << std::endl;
        abort();
        // results.push_back(concept_id);
    }

    std::vector<std::string> final_results;

    for (const auto result : results) {
        ConceptInfo result_info = table.get_info(result);

        if (result_info.vocabulary_id == "Condition Type") {
            std::string final = absl::Substitute("$0/$1", result_info.concept_class_id,
                                                        result_info.concept_code);

            final_results.push_back(final);
        } else {
            std::string final = absl::Substitute("$0/$1", result_info.vocabulary_id,
                                                        result_info.concept_code);

            final_results.push_back(final);
        }

    }

    return final_results;
}

class Cleaner {
   public:
    Cleaner(boost::filesystem::path concept_dir, const char* path)
        : reader(path, false), iterator(reader.iter()) {
        std::cout << "Loaded1" << std::endl;

        ConceptTable concepts = construct_concept_table(concept_dir);

        std::cout << "Got concept table " << std::endl;

        patient_ids = reader.get_patient_ids();
        original_patient_ids = reader.get_original_patient_ids();
        current_index = 0;

        {
            TermDictionary temp_dictionary;
            std::vector<std::pair<std::string, uint32_t>> items =
                reader.get_dictionary().decompose();

            remap_dict.reserve(items.size());

            absl::flat_hash_map<std::string, uint32_t> lost_counts;

            GEMMapper gem("/share/pi/nigam/ethanid/ICDGEM");
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

int main() {
    boost::filesystem::path root(
        "/share/pi/nigam/ethanid/starr_omop_cdm5_latest_extract");

    boost::filesystem::path raw_extract = root / "raw";

    boost::filesystem::path final_extract = root / "final" / "extract.db";

    boost::filesystem::path concept_location = root / "source";

    write_timeline(final_extract.c_str(),
                   Cleaner(concept_location, raw_extract.c_str()));
}