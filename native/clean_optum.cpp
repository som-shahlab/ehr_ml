#include <iostream>

#include "gem.h"
#include "reader.h"
#include "rxnorm.h"
#include "umls.h"
#include "writer.h"

std::vector<std::string> normalize(std::string input_code, const UMLS& umls,
                                   const RxNorm& rxnorm, const GEMMapper& gem) {
    std::vector<std::string_view> parts = absl::StrSplit(input_code, "/");

    std::string_view type = parts[0];

    if (input_code == "") {
        return {};
    } else if (type == "Product") {
        return {input_code};
    } else if (type == "Gender") {
        return {input_code};
    } else if (type == "MainProc") {
        if (parts[1] == "00000") {
            // Remove this junk
            return {};
        }

        if (parts[1][0] == 'J') {
            // J codes need special handling as they are actually medications
            auto codes = rxnorm.get_atc_codes("HCPCS", std::string(parts[1]));
            if (codes.size() != 0) {
                return codes;
            } else {
                // Keep it even though it's a sorta BS code.
                return {absl::Substitute("HCPCS/$0", parts[1])};
            }
        } else {
            std::string candidate;
            if (isdigit(parts[1][0])) {
                candidate = "CPT";
            } else {
                candidate = "HCPCS";
            }

            if (umls.get_aui(candidate, std::string(parts[1]))) {
                return {absl::Substitute("$0/$1", candidate, parts[1])};
            }

            // Even though we couldn't map it, we might as well just keep it.
            // We don't really rely on the CPT hierarchy that much ...
            return {absl::Substitute("$0/$1", candidate, parts[1])};
        }
    } else if (type == "Lab") {
        if (parts[1] == "UNLOINC") {
            return {};
        }

        if (umls.get_aui("LNC", std::string(parts[1]))) {
            return {absl::Substitute("LNC/$0", parts[1])};
        };
        return {input_code};
    } else if (type == "Drug") {
        auto drug_codes = rxnorm.get_atc_codes("NDC", std::string(parts[1]));

        if (drug_codes.size() != 0) {
            return drug_codes;
        }

        // We might as well keep the NDC
        return {input_code};
    } else if (type == "Zip") {
        return {input_code};
    } else if (type == "Drg") {
        return {input_code};
    } else if (type == "Diag" || type == "Proc") {
        std::vector<std::string> codes;

        if (parts[2] == "V9999") {
            // Get rid of this trash
            return {};
        }

        if (parts[1] == "ICD9") {
            // Need to be converted
            if (type == "Diag") {
                codes = gem.map_diag(parts[2]);

                if (codes.size() == 0) {
                    // Try to recover, add a zero (lol)
                    codes = gem.map_diag(absl::Substitute("$00", parts[2]));
                }

                if (codes.size() == 0) {
                    // Try to recover, add a nine (lol)
                    codes = gem.map_diag(absl::Substitute("$09", parts[2]));
                }

                if (codes.size() == 0) {
                    // Try to recover, add a one (lol)
                    codes = gem.map_diag(absl::Substitute("$01", parts[2]));
                }
            } else {
                codes = gem.map_proc(parts[2]);
            }
        } else {
            if (type == "Diag" && parts[2].size() > 3) {
                codes = {absl::Substitute("$0.$1", parts[2].substr(0, 3),
                                          parts[2].substr(3))};
            } else {
                codes = {std::string(parts[2])};
            }
        }

        std::string candidate;

        if (type == "Diag") {
            candidate = "ICD10CM";
        } else {
            candidate = "ICD10PCS";
        }

        std::vector<std::string> results;

        for (const auto& code : codes) {
            if (umls.get_aui(candidate, code)) {
                results.push_back(absl::Substitute("$0/$1", candidate, code));
            }
        }

        if (results.size() != 0) {
            return results;
        }

        // Might as well keep the original if we have to ...
        return {input_code};
    } else {
        std::cout << "unknown type " << type << std::endl;
        abort();
    }
}

class Cleaner {
   public:
    Cleaner(const char* path) : reader(path, false), iterator(reader.iter()) {
        std::cout << "Loaded" << std::endl;

        patient_ids = reader.get_patient_ids();
        original_patient_ids = reader.get_original_patient_ids();
        current_index = 0;

        {
            TermDictionary temp_dictionary;
            std::vector<std::pair<std::string, uint32_t>> items =
                reader.get_dictionary().decompose();

            remap_dict.reserve(items.size());

            absl::flat_hash_map<std::string, uint32_t> lost_counts;

            UMLS umls("/share/pi/nigam/ethanid/UMLS");
            RxNorm rxnorm;
            GEMMapper gem("/share/pi/nigam/ethanid/ICDGEM");

            uint32_t total_lost = 0;

            for (const auto& entry : items) {
                std::vector<std::string> terms =
                    normalize(entry.first, umls, rxnorm, gem);
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
    std::cout << "Hello world" << std::endl;

    std::string_view path = "/share/pi/nigam/secure/optum/ehr_ml/optum_v1";

    std::string write_path = absl::Substitute("$0/$1", path, "clean");
    std::string read_path = absl::Substitute("$0/$1", path, "temp");

    write_timeline(write_path.c_str(), Cleaner(read_path.c_str()));
}