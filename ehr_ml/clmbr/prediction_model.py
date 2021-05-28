import os
import tqdm
import torch
import random
import numpy as np

from torch import nn
from collections import defaultdict

from .dataset import DataLoader
from .rnn_model import PatientRNN
from .sequential_task import SequentialTask
from .labeler_task import LabelerTask
from .doctorai_task import DoctorAITask
from .utils import read_config, read_info, device_from_config

from ..extension.clmbr import PatientTimelineDataset
from ..labeler import SavedLabeler

from typing import Union, List, Tuple, Optional


class CLMBR(nn.Module):
    """
    Encapsulates a model that can encode a timeline, and a module that 
    defines some task.  Examples are PatientRNN and SequentialTask, 
    respectively. These two parts are kept separate b/c for the most
    part we will be using the former as an encoder, and the SequentialTask
    is simply some auxiliary task we are going to use to provide supervision. 
    For our target tasks, we are using the results of compute_embedding
    to run the former part without the auxiliary task machinery.  
    Note that this class doesn't need to know a lot of details about 
    codes vs terms, etc. 
    """

    def __init__(self, config, info, for_labeler=False):
        super().__init__()
        self.config = config
        self.info = info
        self.timeline_model = PatientRNN(config, info)
        if for_labeler:
            self.labeler_module = LabelerTask(config, info)
        else:
            if config.get("doctorai"):
                self.doctorai_module = DoctorAITask(config, info)
            else:
                self.task_module = SequentialTask(
                    config,
                    info,
                    weight=self.timeline_model.input_code_embedding.weight,
                    weight1=self.timeline_model.input_code_embedding1.weight,
                )

    def compute_embedding(self, code_ontology, patient):
        rnn_input = PatientRNN.convert_to_rnn_input(
            self.info, code_ontology, [patient]
        )
        return self.compute_embedding_batch(rnn_input)[0, :, :]

    def compute_embedding_batch(self, rnn_input):
        with torch.no_grad():
            self.eval()
            return self.timeline_model(rnn_input)

    def forward(self, batch):
        rnn_input = batch["rnn"]
        if "task" in batch:
            task_input = batch["task"]
        elif "doctorai" in batch:
            doctorai_input = batch["doctorai"]
        elif "survival" in batch:
            survival_input = batch["survival"]
        elif "labeler" in batch:
            labeler_input = batch["labeler"]

        rnn_output = self.timeline_model(batch["rnn"])
        outputs = dict()

        if "task" in batch:
            values, loss = self.task_module(rnn_output, batch["task"])
        elif "doctorai" in batch:
            values, loss = self.doctorai_module(rnn_output, batch["doctorai"])
        elif "labeler" in batch:
            values, loss = self.labeler_module(rnn_output, batch["labeler"])
        else:
            raise ValueError("Could not find target in batch")

        outputs["rnn"] = rnn_output
        outputs["values"] = values
        outputs["loss"] = loss
        return outputs

    def featurize_patients(
        self,
        extract_dir: str,
        patient_ids: Union[List[str], np.array],
        day_offsets: Union[List[str], np.array],
    ) -> np.array:
        """
        Read info and configuration from a pretrained model dir to load a pretrained CLMBR model
        """
        config = self.config
        dummy_labels = [0 for _ in patient_ids]
        data = (dummy_labels, patient_ids, day_offsets)

        dataset = PatientTimelineDataset(
            os.path.join(extract_dir, "extract.db"),
            os.path.join(extract_dir, "ontology.db"),
            os.path.join(config["model_dir"], "info.json"),
            data,
            data,
        )

        patient_id_to_info = defaultdict(dict)
        for i, (pid, index) in enumerate(zip(patient_ids, day_offsets)):
            patient_id_to_info[pid][index] = i

        patient_representations = np.zeros((len(patient_ids), config["size"]))
        with DataLoader(
            dataset,
            threshold=config["num_first"],
            is_val=True,
            batch_size=config["eval_batch_size"],
            seed=random.randint(0, 100000),
        ) as batches:
            pbar = tqdm.tqdm(total=dataset.num_batches(config["eval_batch_size"], True))
            pbar.set_description("Computing patient representations")
            for batch in batches:
                with torch.no_grad():
                    embeddings = (
                        self.compute_embedding_batch(batch["rnn"]).cpu().numpy()
                    )
                    for i, patient_id in enumerate(batch["pid"]):
                        for index, target_id in patient_id_to_info[
                            patient_id
                        ].items():
                            patient_representations[target_id, :] = embeddings[
                                i, index, :
                            ]
                    pbar.update(1)

        return patient_representations

    def featurize_patients_w_labels(
        self, extract_dir: str, l: SavedLabeler
    ) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Featurize patients using the given model and labeler.
        The result is a numpy array aligned with l.get_labeler_data().
        This function will use the GPU if it is available.
        """
        data = l.get_label_data()

        labels, patient_ids, patient_indices = data

        patient_representations = self.featurize_patients(
            extract_dir, patient_ids, patient_indices
        )

        return patient_representations, labels, patient_ids, patient_indices

    @classmethod
    def from_pretrained(
        cls, model_dir: str, device: Optional[torch.device] = None
    ):
        config = read_config(os.path.join(model_dir, "config.json"))
        info = read_info(os.path.join(model_dir, "info.json"))
        model = cls(config, info)
        model_data = torch.load(
            os.path.join(model_dir, "best"), map_location="cpu"
        )
        model.load_state_dict(model_data)
        if device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        model = model.to(device)
        return model
