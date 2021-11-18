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
#include "csv.h"
#include "parse_utils.h"
#include "writer.h"

const char* location = "/local-scratch/nigam/secure/optum/optum_v8_raw/zip5/";

struct RawPatientRecord {
    uint64_t person_id;
    std::optional<absl::CivilDay> birth_date;
    std::vector<std::pair<absl::CivilDay, uint32_t>> observations;
    std::vector<std::pair<absl::CivilDay, std::pair<uint32_t, uint32_t>>>
        observations_with_values;
    std::vector<std::pair<std::array<char, 10>, uint32_t>> claim_observations;
    std::vector<uint32_t> birth_observations;
    std::vector<std::pair<std::array<char, 10>, absl::CivilDay>> claims;
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

std::array<char, 10> extract_claim_id(std::string_view row) {
    if (row.size() > 10) {
        std::cout << "Got a claim id that's more than 10 characters? " << row
                  << std::endl;
        abort();
    }

    std::array<char, 10> result;
    std::fill(std::begin(result), std::end(result), ' ');

    std::copy(std::begin(row), std::end(row), std::begin(result));

    return result;
}

class Converter {
   public:
    std::string_view get_file() const;
    std::vector<std::string_view> get_columns() const;

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<std::string_view>& row) const;
};

class ClaimItemConverter : public Converter {
   public:
    ClaimItemConverter(std::string f, std::string code_f, std::string cate)
        : filename(f), code_field(code_f), category(cate) {}

    std::string_view get_file() const { return filename; }

    std::vector<std::string_view> get_columns() const {
        return {"PATID", "CLMID", code_field, "ICD_FLAG"};
    }

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<std::string_view>& row) const {
        auto claim_id = extract_claim_id(row[1]);

        std::string code_type;

        if (row[2] == "V9999") {
            return;
        }

        if (row[3] == "9") {
            code_type = "ICD9";
        } else if (row[3] == "10") {
            code_type = "ICD10";
        }

        uint32_t code = meta.dictionary.map_or_add(
            absl::Substitute("$0/$1/$2", category, code_type, row[2]));

        patient_record.claim_observations.push_back(
            std::make_pair(claim_id, code));
    }

   private:
    std::string filename;
    std::string code_field;
    std::string category;
};

class ClaimConverter : public Converter {
   public:
    ClaimConverter(std::string f) : filename(f) {}

    std::string_view get_file() const { return filename; }

    std::vector<std::string_view> get_columns() const {
        return {"PATID", "FST_DT", "CLMID", "NDC", "PROC_CD", "DRG"};
    }

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<std::string_view>& row) const {
        std::string_view date_str;
        if (row[1] != "") {
            date_str = row[1];
        } else {
            std::cout << "Got a claim with no valid date? " << filename << " "
                      << row[2] << std::endl;
            abort();
        }
        auto day = parse_date(date_str);

        auto claim_id = extract_claim_id(row[2]);
        patient_record.claims.push_back(std::make_pair(claim_id, day));

        if (row[3] != "" && row[3] != "1" && row[3] != "UNK" && row[3] != "0" &&
            row[3] != "NONE") {
            uint32_t drug =
                meta.dictionary.map_or_add(absl::Substitute("Drug/$0", row[3]));
            patient_record.observations.push_back(std::make_pair(day, drug));
        }

        if (row[4] != "" && row[4] != "00000") {
            uint32_t proc = meta.dictionary.map_or_add(
                absl::Substitute("MainProc/$0", row[4]));
            patient_record.observations.push_back(std::make_pair(day, proc));
        }

        if (row[5] != "" && row[5] != "000") {
            uint32_t drg =
                meta.dictionary.map_or_add(absl::Substitute("Drg/$0", row[5]));
            patient_record.observations.push_back(std::make_pair(day, drg));
        }
    }

   private:
    std::string filename;
};

class DetailMemberConverter : public Converter {
   public:
    std::string_view get_file() const { return "zip5_mbr_enroll.txt.gz"; }

    std::vector<std::string_view> get_columns() const {
        return {"PATID", "ELIGEFF", "PRODUCT", "ZIPCODE_5", "YRDOB", "GDR_CD"};
    }

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<std::string_view>& row) const {
        int year_of_birth;
        attempt_parse_or_die(row[4], year_of_birth);

        absl::CivilDay birth_date(year_of_birth, 1, 1);

        if (!patient_record.birth_date ||
            *patient_record.birth_date > birth_date) {
            patient_record.birth_date = birth_date;
        }

        uint32_t gender =
            meta.dictionary.map_or_add(absl::Substitute("Gender/$0", row[5]));
        patient_record.birth_observations.push_back(gender);

        auto day = parse_date(row[1]);

        uint32_t prod =
            meta.dictionary.map_or_add(absl::Substitute("Product/$0", row[2]));
        uint32_t zip =
            meta.dictionary.map_or_add(absl::Substitute("Zip/$0", row[3]));

        patient_record.observations.push_back(std::make_pair(day, prod));
        patient_record.observations.push_back(std::make_pair(day, zip));
    }
};

class RxConverter : public Converter {
   public:
    RxConverter(std::string f) : filename(f) {}

    std::string_view get_file() const { return filename; }

    std::vector<std::string_view> get_columns() const {
        return {"PATID", "FILL_DT", "NDC"};
    }

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<std::string_view>& row) const {
        auto day = parse_date(row[1]);

        if (row[1] == "" || row[1] == "1" || row[1] == "UNK" || row[1] == "0") {
            return;
        }

        uint32_t drug =
            meta.dictionary.map_or_add(absl::Substitute("Drug/$0", row[2]));
        patient_record.observations.push_back(std::make_pair(day, drug));
    }

   private:
    std::string filename;
};

class LabConverter : public Converter {
   public:
    LabConverter(std::string f) : filename(f) {}

    std::string_view get_file() const { return filename; }

    std::vector<std::string_view> get_columns() const {
        return {"PATID", "FST_DT", "LOINC_CD", "RSLT_NBR", "RSLT_TXT"};
    }

    void augment_day(Metadata& meta, RawPatientRecord& patient_record,
                     const std::vector<std::string_view>& row) const {
        auto day = parse_date(row[1]);

        if (row[2] == "UNLOINC" || row[2] == "" || row[2] == "PENDING") {
            return;
        }
        uint32_t lab =
            meta.dictionary.map_or_add(absl::Substitute("Lab/$0", row[2]));

        std::string value;

        if (row[4] != "") {
            value = row[4];
        } else if (row[3] != "") {
            value = row[3];
        } else {
            return;
        }

        ObservationWithValue obs;
        obs.code = lab;

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

   private:
    std::string filename;
};

using QueueItem = std::variant<RawPatientRecord, Metadata>;
using Queue = BlockingQueue<QueueItem>;

template <typename C>
void run_converter(C converter, Queue& queue) {
    Metadata meta;

    std::string_view filename = converter.get_file();

    std::cout << absl::Substitute("Starting to work on $0\n", filename);

    std::string full_filename =
        absl::Substitute("$0/$1", location, converter.get_file());

    size_t num_rows = 0;

    RawPatientRecord current_record;
    current_record.person_id = 0;

    csv_iterator(full_filename.c_str(), converter.get_columns(), '|', {}, true,
                 [&](const auto& row) {
                     num_rows++;

                     if (num_rows % 100000000 == 0) {
                         std::cout << absl::Substitute(
                             "Processed $0 rows for $1\n", num_rows, filename);
                     }

                     uint64_t person_id;
                     attempt_parse_or_die(row[0], person_id);

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

    std::cout << absl::Substitute("Done working on $0\n", filename);

    queue.wait_enqueue({std::move(meta)});
}

template <typename C>
std::pair<std::thread, std::shared_ptr<Queue>> generate_converter_thread(
    C converter) {
    std::shared_ptr<Queue> queue =
        std::make_shared<Queue>(10000);  // Ten thousand patient records
    std::thread thread([converter, queue]() {
        std::string_view thread_name = converter.get_file();
        thread_name.remove_prefix(sizeof("zip5_") - 1);
        thread_name = thread_name.substr(0, 15);
        std::string name_copy(std::begin(thread_name), std::end(thread_name));
        int error = pthread_setname_np(pthread_self(), name_copy.c_str());
        if (error != 0) {
            std::cout << "Could not set thread name to " << thread_name << " "
                      << error << std::endl;
            abort();
        }
        run_converter(std::move(converter), *queue);
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

                    for (const auto& obs : record.claim_observations) {
                        total_record.claim_observations.push_back(
                            std::make_pair(obs.first, offset(obs.second)));
                    }

                    for (const auto& obs : record.claims) {
                        total_record.claims.push_back(
                            std::make_pair(obs.first, obs.second));

                        if (obs.second < total_record.birth_date) {
                            total_record.birth_date = obs.second;
                        }
                    }

                    converter_threads[index].second->wait_dequeue(queue_item);
                    contributing_indexes.push_back(index);

                    std::push_heap(std::begin(heap), std::end(heap));
                }

                std::sort(std::begin(total_record.claim_observations),
                          std::end(total_record.claim_observations));
                std::sort(std::begin(total_record.claims),
                          std::end(total_record.claims));

                auto claim_iter = std::begin(total_record.claims);

                for (const auto& claim_obs : total_record.claim_observations) {
                    std::string_view claim_name(claim_obs.first.data(),
                                                claim_obs.first.size());

                    claim_iter =
                        std::find_if(claim_iter, std::end(total_record.claims),
                                     [&](const auto& claim) {
                                         return claim.first == claim_obs.first;
                                     });

                    if (claim_iter == std::end(total_record.claims)) {
                        std::cout << "Could not find claim??? "
                                  << total_record.person_id << " " << claim_name
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
                        abort();
                    }

                    while (
                        ((claim_iter + 1) != std::end(total_record.claims)) &&
                        (claim_iter + 1)->first == claim_obs.first) {
                        // if ((claim_iter + 1)->second != claim_iter->second) {
                        //     std::cout<<claim_name << " Lol date change " <<
                        //     claim_iter->second << " " << (claim_iter +
                        //     1)->second << std::endl;
                        // }
                        claim_iter++;
                    }

                    total_record.observations.push_back(
                        std::make_pair(claim_iter->second, claim_obs.second));
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

std::vector<std::string> glob(const std::string& pattern) {
    using namespace std;

    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if (return_value != 0) {
        globfree(&glob_result);
        stringstream ss;
        ss << "glob() failed with return_value " << return_value << endl;
        throw std::runtime_error(ss.str());
    }

    // collect all the filenames into a std::list<std::string>
    vector<string> filenames;
    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
        filenames.push_back(string(glob_result.gl_pathv[i]));
    }

    // cleanup
    globfree(&glob_result);

    // done
    return filenames;
}

int main() {
    std::string root_directory = "/share/pi/nigam/secure/optum/ehr_ml/optum_v2";

    int error = mkdir(root_directory.c_str(), 0700);

    if (error == -1) {
        std::cout << absl::Substitute(
            "Could not make result directory $0, got error $1\n",
            root_directory, std::strerror(errno));
        exit(-1);
    }

    std::string temp_path = absl::Substitute("$0/temp", root_directory);

    std::vector<std::pair<std::thread, std::shared_ptr<Queue>>>
        converter_threads;

    auto add = [&](auto c) {
        converter_threads.push_back(generate_converter_thread(c));
    };

    add(DetailMemberConverter());

    auto globhelper = [&](std::string_view pattern) {
        auto full_pattern = absl::Substitute("$0/$1", location, pattern);

        std::vector<std::string> results;
        for (const auto& file : glob(full_pattern)) {
            std::vector<std::string> parts = absl::StrSplit(file, "/");
            results.push_back(parts.back());
            // break;
        }

        return results;
    };

    for (const auto& f : globhelper("zip5_m20*")) {
        add(ClaimConverter(f));
    }
    for (const auto& f : globhelper("zip5_diag20*")) {
        add(ClaimItemConverter(f, "DIAG", "Diag"));
    }
    for (const auto& f : globhelper("zip5_proc20*")) {
        add(ClaimItemConverter(f, "PROC", "Proc"));
    }
    for (const auto& f : globhelper("zip5_lr20*")) {
        add(LabConverter(f));
    }
    for (const auto& f : globhelper("zip5_r20*")) {
        add(RxConverter(f));
    }

    write_timeline(temp_path.c_str(), Merger(std::move(converter_threads)));
}
