from typing import Any, Mapping

import torch
from torch import nn

from .labeler_task import LabelerTask
from .rnn_model import PatientRNN
from .sequential_task import SequentialTask


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

    def __init__(
        self,
        config: Mapping[str, Any],
        info: Mapping[str, Any],
        use_cuda: bool,
        for_labeler: bool = False,
    ):
        super().__init__()
        self.config = config
        self.info = info
        self.timeline_model = PatientRNN(config, info)
        if for_labeler:
            self.labeler_module = LabelerTask(config, info)
        else:
            self.task_module = SequentialTask(config, info)
        self.use_cuda = use_cuda

    def compute_embedding_batch(self, rnn_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.eval()
            return self.timeline_model(rnn_input)

        raise RuntimeError("Should never reach here?")

    def forward(self, batch: Mapping[str, Any]) -> Any:  # type: ignore
        rnn_input = batch["rnn"]
        if "task" in batch:
            task_input = batch["task"]
        elif "survival" in batch:
            survival_input = batch["survival"]
        elif "labeler" in batch:
            labeler_input = batch["labeler"]

        rnn_output = self.timeline_model(batch["rnn"])

        if "task" in batch:
            return self.task_module(rnn_output, batch["task"])
        elif "labeler" in batch:
            return self.labeler_module(rnn_output, batch["labeler"])
        else:
            raise ValueError("Could not find target in batch")

    @classmethod
    def finalize_data(
        cls,
        config: Mapping[str, Any],
        info: Mapping[str, Any],
        device: torch.device,
        batch: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        resulting_batch = {}

        resulting_batch["pid"] = batch["pid"].tolist()
        resulting_batch["day_index"] = batch["day_index"].tolist()
        resulting_batch["rnn"] = PatientRNN.finalize_data(
            config, info, device, batch["rnn"]
        )
        if "task" in batch:
            resulting_batch["task"] = SequentialTask.finalize_data(
                config, info, device, batch["task"]
            )
        if "labeler" in batch:
            resulting_batch["labeler"] = LabelerTask.finalize_data(
                config, info, device, batch["labeler"]
            )

        return resulting_batch
