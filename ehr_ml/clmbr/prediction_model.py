import torch
from torch import nn

from .rnn_model import PatientRNN
from .sequential_task import SequentialTask
from .labeler_task import LabelerTask
from .doctorai_task import DoctorAITask


class PredictionModel(nn.Module):
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

        if "task" in batch:
            return self.task_module(rnn_output, batch["task"])
        elif "doctorai" in batch:
            return self.doctorai_module(rnn_output, batch["doctorai"])
        elif "labeler" in batch:
            return self.labeler_module(rnn_output, batch["labeler"])
        else:
            raise ValueError("Could not find target in batch")
