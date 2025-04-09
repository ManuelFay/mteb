from __future__ import annotations

import asyncio
import base64
import os
from functools import partial
from io import BytesIO
from typing import Any, List

import aiohttp
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta


def illuin_v_api_loader(**kwargs):
    model_name = kwargs.get("model_name", "BiQwen2")

    class IlluinAPIModelWrapper:
        def __init__(
            self,
            model_name: str,
            **kwargs: Any,
        ):
            """Wrapper for Illuin API embedding model"""
            self.model_name = model_name
            self.url = model_name
            self.HEADERS = {
                "Accept": "application/json",
                "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
                "Content-Type": "application/json",
            }


        @staticmethod
        def convert_image_to_base64(image: Image.Image) -> str:
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        async def post_images(self, session: aiohttp.ClientSession, encoded_images: List[str]):
            payload = {"inputs": {"images": encoded_images}}
            async with session.post(self.url, headers=self.HEADERS, json=payload) as response:
                return await response.json()

        async def post_queries(self, session: aiohttp.ClientSession, queries: List[str]):
            payload = {"inputs": {"queries": queries}}
            async with session.post(self.url, headers=self.HEADERS, json=payload) as response:
                return await response.json()

        async def call_api_queries(self, queries: List[str]):
            embeddings = []
            semaphore = asyncio.Semaphore(16)
            async with aiohttp.ClientSession() as session:

                async def sem_post(batch):
                    async with semaphore:
                        return await self.post_queries(session, batch)

                tasks = [asyncio.create_task(sem_post([batch])) for batch in queries]

                # ORDER-PRESERVING
                results = await tqdm_asyncio.gather(*tasks, desc="Query batches")

                for result in results:
                    embeddings.extend(result.get("embeddings", []))

            return embeddings

        async def call_api_images(self, images_b64: List[str]):
            embeddings = []
            semaphore = asyncio.Semaphore(16)

            async with aiohttp.ClientSession() as session:

                async def sem_post(batch):
                    async with semaphore:
                        return await self.post_images(session, batch)

                tasks = [asyncio.create_task(sem_post([batch])) for batch in images_b64]

                # ORDER-PRESERVING
                results = await tqdm_asyncio.gather(*tasks, desc="Doc batches")

                for result in results:
                    embeddings.extend(result.get("embeddings", []))

            return embeddings

        def forward_queries(self, queries: List[str]) -> torch.Tensor:
            response = asyncio.run(self.call_api_queries(queries))
            return response

        def forward_passages(self, passages: List[Image.Image]) -> torch.Tensor:
            response = asyncio.run(self.call_api_images([self.convert_image_to_base64(doc) for doc in passages]))
            return response

        def get_text_embeddings(
            self,
            texts: list[str],
            *,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            batch_size: int = 32,
            **kwargs: Any,
        ):
            all_text_embeddings = self.forward_queries(texts)
            all_text_embeddings = torch.cat([torch.as_tensor(t) for t in all_text_embeddings], dim=0)
            return all_text_embeddings

        def get_image_embeddings(
            self,
            images: list[Image.Image] | DataLoader,
            *,
            task_name: str | None = None,
            prompt_type: PromptType | None = None,
            batch_size: int = 32,
            **kwargs: Any,
        ):
            all_image_embeddings = []

            if isinstance(images, DataLoader):
                for batch_images in tqdm(images):
                    batch_images = [torchvision.transforms.functional.to_pil_image(img) for img in batch_images]
                    all_image_embeddings.extend(self.forward_passages(batch_images))
            else:
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    all_image_embeddings.extend(self.forward_passages(batch_images))
            all_image_embeddings = torch.cat([torch.as_tensor(t) for t in all_image_embeddings], dim=0)
            return all_image_embeddings

        def calculate_probs(self, text_embeddings, image_embeddings):
            text_embeddings = text_embeddings / text_embeddings.norm(
                dim=-1, keepdim=True
            )
            image_embeddings = image_embeddings / image_embeddings.norm(
                dim=-1, keepdim=True
            )
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

    return IlluinAPIModelWrapper(**kwargs)


vidore_biqwen2 = ModelMeta(
    loader=partial(illuin_v_api_loader, model_name="https://es2rk709av11wzkm.us-east-1.aws.endpoints.huggingface.cloud"),
    name="vidore/biqwen2-v0.1",
    languages=[],  # Unknown, but support >100 languages
    revision="1",
    release_date="2024-10-24",
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=1024,
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
