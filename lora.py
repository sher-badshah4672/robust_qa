import inspect
import os
import warnings
from typing import Dict, List, Optional, Set, Tuple, Union
from peft.peft_model import PeftModel
from peft.utils import (
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    WEIGHTS_NAME,
    PeftConfig,
    PeftType,
    PromptLearningConfig,
    TaskType,
    _set_trainable,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    shift_tokens_right,
)
import torch

class PeftModeForQuestionAnswering(PeftModel):
    def __init__(self, model, peft_config: PeftConfig):
        super().__init__(model, peft_config)
        self.modules_to_save = ["classifier", "score"]

        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not isinstance(self.peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                start_positions=start_positions,
                end_positions=end_positions,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        else:
            print(f'PromptLearning currently isnt supported for QuestionAnswering task')