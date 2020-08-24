from ehr_ml.labeler import Label, Labeler, InpatientAdmissionHelper, LabelType
import ehr_ml.timeline
import ehr_ml.index

from typing import List, Optional, Set

class LongAdmissionLabeler(Labeler):
    """
    The inpatient labeler predicts whether or not a patient will be admitted for a long time (defined 
    as greater than 7 days).
    The prediction time is before they get admitted
    """
    def __init__(self, timelines: ehr_ml.timeline.TimelineReader, ind: ehr_ml.index.Index):
        self.admission_helper = InpatientAdmissionHelper(timelines)

        self.all_patient_ids = self.admission_helper.get_all_patient_ids(ind)

    def label(self, patient: ehr_ml.timeline.Patient) -> List[Label]:
        admissions = self.admission_helper.get_inpatient_admissions(patient)

        labels = []

        current_admission_index = 0

        for i, day in enumerate(patient.days):
            if current_admission_index >= len(admissions):
                continue
            current_admission = admissions[current_admission_index]

            assert day.age <= current_admission.start_age

            if day.age == current_admission.start_age:
                current_admission_index += 1

                long_admission = current_admission.end_age - current_admission.start_age >= 3

                if i != 0:
                    labels.append(Label(day_index=i - 1, is_positive=long_admission))

        return labels
    
    def get_all_patient_ids(self) -> Optional[Set[int]]:
        return self.all_patient_ids
    
    def get_labeler_type(self) -> LabelType:
        return "binary"
