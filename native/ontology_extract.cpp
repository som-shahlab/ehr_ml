#include <set>

#include "concept.h"
#include "constdb.h"
#include "csv.h"
#include "reader.h"
#include "umls_util.h"

const char* cdm_location = "/scratch/labs/shahlab/OPTUM_CDMv5/Optum_CDM";

const std::string* try_to_recover(
    const absl::flat_hash_map<std::pair<std::string, std::string>, std::string>&
        code_to_aui_map,
    const ConceptTable& table, uint32_t concept_id) {
    ConceptInfo info = table.get_info(concept_id);

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
            ConceptInfo c_info = table.get_info(candidate);
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
        return nullptr;
    } else if (new_id_candidates.size() > 1) {
        std::cout << "Odd " << info.vocabulary_id << " " << info.concept_code
                  << " " << concept_id << " " << new_id_candidates.size()
                  << std::endl;
        return nullptr;
    } else {
        const auto& info = table.get_info(new_id_candidates[0]);

        for (std::string terminology :
             map_terminology_type(info.vocabulary_id)) {
            auto aui_iter = code_to_aui_map.find(
                std::make_pair(terminology, info.concept_code));

            if (aui_iter != std::end(code_to_aui_map)) {
                return &(aui_iter->second);
            }
        }

        return try_to_recover(code_to_aui_map, table, new_id_candidates[0]);
    }
}

std::vector<uint32_t> compute_subwords(
    std::string aui,
    const absl::flat_hash_map<std::string, std::pair<std::string, std::string>>&
        aui_to_code_map,
    const absl::flat_hash_map<std::string, std::vector<std::string>>&
        aui_to_parents_map,
    absl::flat_hash_map<std::string, std::vector<uint32_t>>&
        aui_to_subwords_map,
    absl::flat_hash_map<uint32_t, std::vector<uint32_t>>& code_to_parents_map,
    TermDictionary& dictionary) {
    if (aui_to_subwords_map.find(aui) == std::end(aui_to_subwords_map)) {
        std::vector<uint32_t> results;
        auto parents_iter = aui_to_parents_map.find(aui);

        auto info_iter = aui_to_code_map.find(aui);

        if (info_iter == std::end(aui_to_code_map)) {
            std::cout << "Could not find " << aui << std::endl;
            abort();
        }

        std::string word = absl::Substitute("$0/$1", info_iter->second.first,
                                            info_iter->second.second);

        uint32_t aui_code = dictionary.map_or_add(word);

        results.push_back(aui_code);

        if (parents_iter != std::end(aui_to_parents_map)) {
            for (const auto& parent_aui : parents_iter->second) {
                auto parent_info_iter = aui_to_code_map.find(parent_aui);

                if (parent_info_iter == std::end(aui_to_code_map)) {
                    std::cout << "Could not find " << parent_aui << std::endl;
                    abort();
                }

                std::string parent_word =
                    absl::Substitute("$0/$1", parent_info_iter->second.first,
                                     parent_info_iter->second.second);

                code_to_parents_map[aui_code].push_back(
                    dictionary.map_or_add(parent_word));

                for (uint32_t subword :
                     compute_subwords(parent_aui, aui_to_code_map,
                                      aui_to_parents_map, aui_to_subwords_map,
                                      code_to_parents_map, dictionary)) {
                    results.push_back(subword);
                }
            }
        }

        if (std::find(std::begin(results), std::end(results),
                      dictionary.map_or_add("SRC/V-SRC")) ==
            std::end(results)) {
            std::cout << "AUI " << aui << " has no root " << std::endl;
            abort();
        }

        std::sort(std::begin(results), std::end(results));
        results.erase(std::unique(std::begin(results), std::end(results)),
                      std::end(results));
        aui_to_subwords_map.insert(std::make_pair(aui, std::move(results)));
    }

    return aui_to_subwords_map.find(aui)->second;
}

int main() {
    std::string target = "/labs/shahlab/projects/ethanid/ehr_ml/native/result7";
    std::string root_path = absl::Substitute("$0_final", target);

    ExtractReader extract(root_path.c_str(), false);

    const TermDictionary& dictionary = extract.get_dictionary();

    auto [words, counts] = dictionary.decompose();

    auto code_to_aui_map = load_code_to_aui_map();

    absl::flat_hash_map<std::string, std::pair<std::string, std::string>>
        aui_to_code_map;
    for (auto& iter : code_to_aui_map) {
        aui_to_code_map.insert(std::make_pair(iter.second, iter.first));
    }

    auto aui_to_parents_map = load_aui_to_parents_map(aui_to_code_map);

    ConceptTable table = construct_concept_table(cdm_location);

    TermDictionary ontology_dictionary;

    std::string ontology_path = absl::Substitute("$0_ontology", target);
    ConstdbWriter ontology_writer(ontology_path.c_str());

    absl::flat_hash_map<std::string, std::vector<uint32_t>> aui_to_subwords_map;
    absl::flat_hash_map<uint32_t, std::vector<uint32_t>> code_to_parents_map;

    std::vector<uint32_t> words_with_subwords;
    std::vector<uint32_t> recorded_date_codes;

    std::set<std::string> code_types;

    std::set<std::string> recorded_date_code_types = {
        "ATC",   "CPT4",    "DRG",      "Gender",
        "HCPCS", "ICD10CM", "ICD10PCS", "LOINC"};

    for (uint32_t i = 0; i < words.size(); i++) {
        const auto& word = words[i];
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

        const std::string* result = nullptr;

        for (std::string terminology : map_terminology_type(parts[0])) {
            auto aui_iter = code_to_aui_map.find(
                std::make_pair(terminology, std::string(parts[1])));

            if (aui_iter != std::end(code_to_aui_map)) {
                result = &(aui_iter->second);
            }
        }

        if (result == nullptr &&
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
            result = try_to_recover(code_to_aui_map, table, concept_id);
        }

        if (result == nullptr) {
            subwords = {ontology_dictionary.map_or_add(
                absl::Substitute("NO_MAP/$0", word))};
        } else {
            subwords = compute_subwords(
                *result, aui_to_code_map, aui_to_parents_map,
                aui_to_subwords_map, code_to_parents_map, ontology_dictionary);
        }

        ontology_writer.add_int(i, (const char*)subwords.data(),
                                subwords.size() * sizeof(uint32_t));
        words_with_subwords.push_back(i);
    }

    for (auto& iter : code_to_parents_map) {
        auto& parent_codes = iter.second;
        std::sort(std::begin(parent_codes), std::end(parent_codes));

        int32_t subword_as_int = iter.first;
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
