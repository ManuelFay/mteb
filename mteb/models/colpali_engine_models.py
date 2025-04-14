from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torchvision
import transformers
from packaging import version
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from colpali_engine.models import BiQwen2, BiQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available


from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta

class ColPaliEngineWrapper:
    def __init__(
        self,
        model_name: str,
        composed_prompt=None,
        **kwargs: Any,
    ):

        self.model_name = model_name
        self.processor = BiQwen2Processor.from_pretrained(model_name)
        self.model = BiQwen2.from_pretrained(
            model_name,
            torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),
            device_map=kwargs.get("device_map", "cuda"),
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()

        if composed_prompt:
            raise NotImplementedError

    def get_text_embeddings(
        self,
        texts: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 8,
        **kwargs: Any,
    ):
        all_text_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                text_inputs = self.processor.process_queries(batch_texts).to(self.model.device)
                text_outputs = self.model(
                    **text_inputs, output_hidden_states=True, return_dict=True
                )
                all_text_embeddings.append(text_outputs.cpu())
        return torch.cat(all_text_embeddings, dim=0)

    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 8,
        **kwargs: Any,
    ):
        all_image_embeddings = []

        with torch.no_grad():
            if isinstance(images, DataLoader):
                for batch_images in tqdm(images):
                    batch_images = [torchvision.transforms.functional.to_pil_image(img) for img in batch_images]
                    img_inputs = self.processor.process_images(batch_images).to(self.model.device)
                    image_outputs = self.model(
                        **img_inputs, output_hidden_states=True, return_dict=True
                    )
                    all_image_embeddings.append(image_outputs.cpu())
            else:
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    img_inputs = self.processor.process_images(batch_images).to(self.model.device)
                    image_outputs = self.model(
                        **img_inputs, output_hidden_states=True, return_dict=True
                    )
                    all_image_embeddings.append(image_outputs.cpu())
            return torch.cat(all_image_embeddings, dim=0)

    def calculate_probs(self, text_embeddings, image_embeddings):
        logits = torch.matmul(image_embeddings, text_embeddings.T)
        probs = (logits * 100).softmax(dim=-1)
        return probs

    def get_fused_embeddings(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image] | DataLoader | None = None,
        fusion_mode="sum",
        **kwargs: Any,
    ):
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        text_embeddings = None
        image_embeddings = None

        if texts is not None:
            text_embeddings = self.get_text_embeddings(texts, **kwargs)

        if images is not None:
            image_embeddings = self.get_image_embeddings(images, **kwargs)

        if text_embeddings is not None and image_embeddings is not None:
            print(
                f"Fusing text and image embeddings with fusion mode: {fusion_mode}"
            )
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            if fusion_mode == "sum":
                fused_embeddings = text_embeddings + image_embeddings
            else:
                # to do: add other fusion mode
                raise ValueError(
                    f"fusion mode {fusion_mode} hasn't been implemented"
                )
            return fused_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings


colpali_engine_models = ModelMeta(
    loader=partial(
        ColPaliEngineWrapper,
        model_name="vidore/biqwen2-v0.1",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ),
    name="vidore/biqwen2-v0.1",
    languages=[],  # Unknown, but support >100 languages
    revision="1",
    release_date="2024-10-24",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=1536,
    license=None,
    similarity_fn_name="cosine",
    framework=[],
    modalities=["image", "text"],
    open_weights=False,
    public_training_code=None,
    public_training_data=None,
    reference="https://huggingface.co/vidore/biqwen2-v0.1",
    use_instructions=False,
    training_datasets=None,
)
