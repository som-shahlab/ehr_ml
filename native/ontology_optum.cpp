#include <set>

#include "constdb.h"
#include "reader.h"
#include "umls.h"

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

std::vector<uint32_t> compute_zip_subwords(
    std::string_view zip,
    absl::flat_hash_map<uint32_t, std::vector<uint32_t>>& code_to_parents_map,
    TermDictionary& dictionary) {
    std::vector<std::string_view> parts = absl::StrSplit(zip, '_');

    // std::cout<<"Got zip " << zip << std::endl;
    // for (const auto& part : parts) {
    //     std::cout<<part << std::endl;
    // }

    std::string_view first_zip = parts[0];  // Arbitrary choice

    std::vector<std::string> vals = {
        "SRC/V-SRC",
        "SRC/ZIP",
        absl::StrCat("ZIP/", first_zip.substr(0, 1), "----"),
        absl::StrCat("ZIP/", first_zip.substr(0, 3), "--"),
        absl::StrCat("ZIP/", first_zip),
    };

    std::vector<uint32_t> result;

    for (size_t i = 0; i < vals.size(); i++) {
        // std::cout<<"Converted to " << vals[i] << std::endl;

        uint32_t next = dictionary.map_or_add(vals[i]);
        if (i != 0) {
            uint32_t last = result.back();
            code_to_parents_map[last].push_back(last);
        }
        result.push_back(next);
    }

    return result;
}

int main() {
    std::string root_path =
        "/share/pi/nigam/secure/optum/ehr_ml/optum_v1/clean";

    ExtractReader extract(root_path.c_str(), false);

    const TermDictionary& dictionary = extract.get_dictionary();

    auto words = dictionary.decompose();

    UMLS umls("/share/pi/nigam/ethanid/UMLS");

    TermDictionary ontology_dictionary;

    std::string ontology_path = absl::Substitute("$0_ontology", root_path);
    ConstdbWriter ontology_writer(ontology_path.c_str());

    absl::flat_hash_map<std::string, std::vector<uint32_t>> aui_to_subwords_map;
    absl::flat_hash_map<uint32_t, std::vector<uint32_t>> code_to_parents_map;

    std::vector<uint32_t> words_with_subwords;
    std::vector<uint32_t> recorded_date_codes;

    std::set<std::string> code_types;

    for (uint32_t i = 0; i < words.size(); i++) {
        const auto& word = words[i].first;
        std::vector<uint32_t> subwords = {};

        std::vector<absl::string_view> parts = absl::StrSplit(word, '/');

        if (parts.size() < 2) {
            std::cout << "Got weird vocab string " << word << std::endl;
            abort();
        }

        code_types.insert(std::string(parts[0]));

        recorded_date_codes.push_back(i);

        if (parts[0] == "Zip") {
            // Zip codes need special handing ...

            subwords = compute_zip_subwords(parts[1], code_to_parents_map,
                                            ontology_dictionary);

        } else {
            auto result =
                umls.get_aui(std::string(parts[0]), std::string(parts[1]));

            if (result) {
                subwords =
                    compute_subwords(*result, umls, aui_to_subwords_map,
                                     code_to_parents_map, ontology_dictionary);
            }

            if (subwords.size() == 0) {
                subwords = {ontology_dictionary.map_or_add(
                    absl::Substitute("NO_MAP/$0", word))};
            }
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
