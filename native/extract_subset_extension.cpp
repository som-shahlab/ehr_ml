#include "extract_subset_extension.h"

#include <sys/stat.h>
#include <assert.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "constdb.h"
#include "reader.h"
#include "writer.h"

class Reader2Writer {
  public:
    Reader2Writer(const char * src_path,
		  float subset_ratio) : reader(src_path, true), iterator(reader.iter()) {
      size_t total_patients = reader.get_patient_ids().size();
      size_t target_num_patients = static_cast<size_t>((float) total_patients * subset_ratio);
      std::cout << "Selecting " << target_num_patients << " out of " << total_patients << " for subset" << std::endl;
      size_t num_processed = 0;
      for (auto patient_id : reader.get_patient_ids()) {
	if (num_processed >= target_num_patients) {
	  break;
	}
	patient_ids.push_back(patient_id);
	num_processed++;
      }
      patient_id_iterator = patient_ids.begin();
    }

    WriterItem operator()() {
      // essentially just iterate through patient_ids
      if (patient_id_iterator == patient_ids.end()) {
	// need to return metadata to conclude write_timeline

	// should I be returning the exact same dictionaries as the original extract?
	Metadata meta;
	meta.dictionary = reader.get_dictionary();
	meta.value_dictionary = reader.get_value_dictionary();
	return meta;
      }
      uint32_t patient_id = *patient_id_iterator;
      patient_id_iterator++;
      PatientRecord final_record;
      final_record.person_id = patient_id;
      iterator.process_patient(patient_id,
			       [&](absl::CivilDay birth_day, uint32_t age,
				   const std::vector<uint32_t>& observations,
				   const std::vector<ObservationWithValue>&
				   observations_with_values) {
				 final_record.birth_date = birth_day;
				 for (uint32_t code : observations) {
				   final_record.observations.push_back(std::make_pair(age, code));
				 }
				 for (ObservationWithValue obswv : observations_with_values) {
				   final_record.observations_with_values.push_back(std::make_pair(age, obswv.encode()));
				 }
			       });
      return final_record;
    }

  ~Reader2Writer() {
    std::cout << "Finished writing subset!" << std::endl;
  }

  private:
    ExtractReader reader;
    ExtractReaderIterator iterator;
    std::vector<uint32_t> patient_ids;
    std::vector<uint32_t>::iterator patient_id_iterator;
    
};
  
void omop_subset_extraction(std::string src_timeline_path,
			    std::string target_timeline_path,
			    float subset_ratio) {

  assert (0.0 < subset_ratio < 1.0);
  
  write_timeline(target_timeline_path.c_str(),
		 Reader2Writer(src_timeline_path.c_str(), subset_ratio));
}

void register_subset_extension(py::module& root) {
  py::module m = root.def_submodule("subset");

  m.def("extract_subset", omop_subset_extraction);
}
