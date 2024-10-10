import torch
import torch.nn.functional as F
import transformers
from peft import get_peft_model
from torch import nn


# patch flashattention into dinov2
class Dinov2SelfAttention(transformers.models.dinov2.modeling_dinov2.Dinov2SelfAttention):
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask=None, output_attentions: bool = False):
        assert not output_attentions
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        context_layer = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, attn_mask=head_mask)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer,)
        return outputs


transformers.models.dinov2.modeling_dinov2.Dinov2SelfAttention = Dinov2SelfAttention


# dinov2 uses clsavg token -> use only cls
class IgnoreAverageTokenLinear(nn.Linear):
    def __call__(self, x):
        cls, _ = x.chunk(chunks=2, dim=1)
        return super().forward(cls)


class Dinov2PeftModel(nn.Module):
    def __init__(
            self,
            input_shape,
            num_outputs,
            identifier,
            peft_config=None,
            head_init="truncnormal2e5",
            pooling="class_token",
    ):
        # facebook/dinov2-small
        # facebook/dinov2-large
        super().__init__()
        assert input_shape == (3, 224, 224), f"expected input_shape (3, 224, 224) but got {input_shape}"
        self.identifier = identifier

        # init model
        model = transformers.Dinov2ForImageClassification.from_pretrained(
            identifier,
            num_labels=num_outputs,
        )
        if head_init == "none":
            pass
        elif head_init == "truncnormal2e5":
            nn.init.trunc_normal_(model.classifier.weight, std=2e-5)
            # bias is automatically set to 0 by hugginface
            # nn.init.zeros_(model.classifier.bias)
        else:
            raise NotImplementedError
        self.dim = model.config.hidden_size

        # init peft
        if peft_config is not None:
            kind = peft_config.pop("kind")
            if kind == "lora_config":
                from peft import LoraConfig
                peft_config = LoraConfig(
                    target_modules=["query", "value", "key", "output.dense", "mlp.fc1", "mlp.fc2"],
                    **peft_config,
                )
                model = get_peft_model(model=model, peft_config=peft_config)
            elif kind == "boft_config":
                from peft import BOFTConfig
                peft_config = BOFTConfig(
                    target_modules=["query", "value", "key", "output.dense", "mlp.fc1", "mlp.fc2"],
                    **peft_config,
                )
                model = get_peft_model(model=model, peft_config=peft_config)
            elif kind == "adalora_config":
                from peft import AdaLoraConfig
                peft_config = AdaLoraConfig(
                    target_modules=["query", "value", "key", "output.dense", "mlp.fc1", "mlp.fc2"],
                    **peft_config,
                )
                model = get_peft_model(model=model, peft_config=peft_config)
            else:
                raise NotImplementedError

        # peft freezes classifier
        if peft_config is not None:
            model.base_model.model.classifier.weight.requires_grad = True
            model.base_model.model.classifier.bias.requires_grad = True

        # huggingface implementation uses clsavg token -> replace with cls token
        if pooling == "class_token":
            if peft_config is None:
                base = model
            else:
                base = model.base_model.model
            old_classifier = base.classifier
            new_classifier = IgnoreAverageTokenLinear(
                old_classifier.in_features // 2,
                old_classifier.out_features,
            )
            new_classifier.weight.data.copy_(old_classifier.weight[:, :old_classifier.in_features // 2])
            new_classifier.bias = old_classifier.bias
            setattr(model, "classifier", new_classifier)
            del old_classifier
        else:
            raise NotImplementedError

        # register module
        self.model = model

    def __str__(self):
        return self.identifier

    def forward(self, x):
        return self.model(x).logits
