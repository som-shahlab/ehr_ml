#ifndef CONCEPT_H_INCLUDED
#define CONCEPT_H_INCLUDED

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/substitute.h"
#include "avro_utils.h"
#include "boost/filesystem.hpp"
#include "parse_utils.h"
#include "csv.h"

struct ConceptInfo {
    std::string vocabulary_id;
    std::string concept_code;
    std::string concept_class_id;
};

struct ConceptRelationshipInfo {
    uint32_t other_concept;
    std::string relationship_id;
};
class ConceptTable {
 public:
  void add_concept(uint32_t concept_id, ConceptInfo info) {
    inverse_lookup.insert(std::make_pair(
        std::make_pair(info.vocabulary_id, info.concept_code), concept_id));
    concepts.insert(std::make_pair(concept_id, info));
  }

  void add_relationship(uint32_t concept_id, ConceptRelationshipInfo info) {
    auto [iter, added] = relationships.insert(
        std::make_pair(concept_id, std::vector<ConceptRelationshipInfo>()));

    (void)added;  // Ignore

    iter->second.push_back(std::move(info));
  }

  void add_ancestor(uint32_t concept_id, uint32_t ancestor) {
    auto [iter, added] =
        ancestors.insert(std::make_pair(concept_id, std::vector<uint32_t>()));

    (void)added;  // Ignore

    iter->second.push_back(ancestor);
  }

  std::optional<ConceptInfo> get_info(uint32_t concept_id) const {
    auto iter = concepts.find(concept_id);

    if (iter == std::end(concepts)) {
      return std::nullopt;
    }

    return iter->second;
  }

  std::vector<ConceptRelationshipInfo> get_relationships(
      uint32_t concept_id) const {
    auto iter = relationships.find(concept_id);

    if (iter == std::end(relationships)) {
      return {};
    } else {
      return iter->second;
    }
  }

  std::vector<uint32_t> get_ancestors(uint32_t concept_id) const {
    auto iter = ancestors.find(concept_id);

    if (iter == std::end(ancestors)) {
      return {};
    } else {
      return iter->second;
    }
  }

  std::optional<uint32_t> get_inverse(std::string vocab,
                                      std::string code) const {
    auto iter = inverse_lookup.find(std::make_pair(vocab, code));

    if (iter == std::end(inverse_lookup)) {
      return {};
    } else {
      return {iter->second};
    }
  }

 private:
  absl::flat_hash_map<std::pair<std::string, std::string>, uint32_t>
      inverse_lookup;
  absl::flat_hash_map<uint32_t, ConceptInfo> concepts;
  absl::flat_hash_map<uint32_t, std::vector<ConceptRelationshipInfo>>
      relationships;
  absl::flat_hash_map<uint32_t, std::vector<uint32_t>> ancestors;
};

bool has_prefix(std::string_view a, std::string_view b) {
    return a.substr(0, b.size()) == b;
}


ConceptTable construct_concept_table_csv(std::string_view location) {
  ConceptTable result;

  std::string concept_file =
      absl::Substitute("$0/$1", location, "concept.csv.gz");

  std::vector<std::string_view> columns = {"concept_id", "vocabulary_id",
                                           "concept_code", "concept_class_id"};

  csv_iterator(concept_file.c_str(), columns, '\t', {}, false,
               [&result](const auto& row) {
                 uint32_t concept_id;
                 attempt_parse_or_die(row[0], concept_id);

                 ConceptInfo info;
                 info.vocabulary_id = row[1];
                 info.concept_code = row[2];
                 info.concept_class_id = row[3];

                 result.add_concept(concept_id, std::move(info));
               });

  std::string concept_file_rel =
      absl::Substitute("$0/$1", location, "concept_relationship.csv.gz");

  std::vector<std::string_view> rel_columns = {"concept_id_1", "concept_id_2",
                                               "relationship_id"};

  csv_iterator(concept_file_rel.c_str(), rel_columns, '\t', {}, false,
               [&result](const auto& row) {
                 uint32_t concept_id;
                 attempt_parse_or_die(row[0], concept_id);
                 uint32_t other_concept;
                 attempt_parse_or_die(row[1], other_concept);

                 ConceptRelationshipInfo info;
                 info.other_concept = other_concept;
                 info.relationship_id = row[2];

                 result.add_relationship(concept_id, std::move(info));
               });

  std::string concept_file_anc =
      absl::Substitute("$0/$1", location, "concept_ancestor.csv.gz");

  std::vector<std::string_view> anc_columns = {"ancestor_concept_id",
                                               "descendant_concept_id"};

  csv_iterator(concept_file_anc.c_str(), anc_columns, '\t', {}, false,
               [&result](const auto& row) {
                 uint32_t ancestor_concept_id;
                 attempt_parse_or_die(row[0], ancestor_concept_id);
                 uint32_t descendant_concept_id;
                 attempt_parse_or_die(row[1], descendant_concept_id);
                 result.add_ancestor(descendant_concept_id,
                                     ancestor_concept_id);
               });

  return result;
}

ConceptTable construct_concept_table_avro(boost::filesystem::path root_directory) {
    ConceptTable result;

    std::vector<std::string_view> concept_columns = {
        "concept_id", "vocabulary_id", "concept_code", "concept_class_id"};

    std::vector<std::string_view> rel_columns = {"concept_id_1", "concept_id_2",
                                                 "relationship_id"};

    std::vector<std::string_view> anc_columns = {"ancestor_concept_id",
                                                 "descendant_concept_id"};

    for (const auto& subfile :
         boost::filesystem::directory_iterator(root_directory)) {
        std::string filename = subfile.path().filename().string();

        std::string_view filename_view(filename);

        if (has_prefix(filename_view, "concept0")) {
            parse_avro_file(subfile, concept_columns,
                            [&result](const auto& row) {
                                uint32_t concept_id = get_long(row[0]);

                                ConceptInfo info;
                                info.vocabulary_id = get_string(row[1]);
                                info.concept_code = get_string(row[2]);
                                info.concept_class_id = get_string(row[3]);

                                result.add_concept(concept_id, std::move(info));
                            });
        } else if (has_prefix(filename_view, "concept_relationship0")) {
            parse_avro_file(subfile, rel_columns, [&result](const auto& row) {
                uint32_t concept_id = get_long(row[0]);
                uint32_t other_concept = get_long(row[1]);

                ConceptRelationshipInfo info;
                info.other_concept = other_concept;
                info.relationship_id = get_string(row[2]);

                result.add_relationship(concept_id, std::move(info));
            });
        } else if (has_prefix(filename_view, "concept_ancestor0")) {
            parse_avro_file(subfile, anc_columns, [&result](const auto& row) {
                uint32_t ancestor_concept_id = get_long(row[0]);
                uint32_t descendant_concept_id = get_long(row[1]);
                result.add_ancestor(descendant_concept_id, ancestor_concept_id);
            });
        }
    }

    return result;
}

#endif