"""LLM adapters for page level text transcription.

Each adapter wraps one LLM provider (a local HuggingFace vision language model
such as Chandra, Gemini, or any OpenAI compatible endpoint) behind a common
interface: it receives a full page image and returns the transcribed text
lines, optionally with a bounding box per line (pixel coordinates of the input
image). New providers only need to subclass ``LLMOCRAdapter`` and register
themselves in ``ADAPTER_REGISTRY``.
"""
import base64
import io
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class LLMTextLine:
    text: str
    # (x1, y1, x2, y2) in pixel coordinates of the image that was sent to the LLM
    bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class LLMPageTranscription:
    lines: List[LLMTextLine]
    raw_response: str = ''


DEFAULT_PROMPT = (
    "You are an expert in transcribing medieval music manuscripts. "
    "The image shows a page of a manuscript with staves of music notation and text lines "
    "(lyrics, headings and paragraphs) between them.\n"
    "Transcribe ONLY the written text lines (do not describe the music notation, do not "
    "transcribe isolated decorated initials/drop capitals as separate lines).\n"
    "Keep the original spelling, do not modernize or expand abbreviations.\n"
    "Return the result as a JSON array, one entry per physical text line in reading order "
    "(top to bottom, left column before right column). Each entry must be an object:\n"
    '{"text": "<transcription of the line>", "bbox": [x1, y1, x2, y2]}\n'
    "where bbox is the bounding box of the text line in pixel coordinates of the given image "
    "(x1,y1 = top left, x2,y2 = bottom right). If you cannot determine a bounding box, use null.\n"
    "The image size is {width}x{height} pixels.\n"
    "Answer with the JSON array only, no additional text."
)


def build_prompt(image: Image.Image, custom_prompt: Optional[str] = None) -> str:
    prompt = custom_prompt if custom_prompt else DEFAULT_PROMPT
    return prompt.replace('{width}', str(image.width)).replace('{height}', str(image.height))


def _scale_bbox(bbox: List[float], image_size: Tuple[int, int]) -> Optional[Tuple[float, float, float, float]]:
    if not bbox or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None

    w, h = image_size
    m = max(x1, y1, x2, y2)
    if m <= 1.5:
        # normalized [0, 1] coordinates
        x1, x2 = x1 * w, x2 * w
        y1, y2 = y1 * h, y2 * h
    elif m <= 1000 and (x2 > w or y2 > h):
        # coordinates exceed the image but fit into [0, 1000]:
        # qwen-vl style [0, 1000] normalized coordinates
        x1, x2 = x1 / 1000 * w, x2 / 1000 * w
        y1, y2 = y1 / 1000 * h, y2 / 1000 * h
    # else: pixel coordinates as requested in the prompt

    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def parse_llm_response(raw: str, image_size: Tuple[int, int]) -> List[LLMTextLine]:
    """Parse the LLM answer into text lines.

    Preferred format is a JSON array of {"text", "bbox"} objects, but plain
    text (one transcription per line) is accepted as fallback.
    """
    text = raw.strip()
    # strip markdown code fences
    fence = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # try to locate a JSON array in the response
    data = None
    start, end = text.find('['), text.rfind(']')
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            data = None

    lines: List[LLMTextLine] = []
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, str):
                if entry.strip():
                    lines.append(LLMTextLine(text=entry.strip()))
            elif isinstance(entry, dict):
                t = str(entry.get('text', '') or '').strip()
                if not t:
                    continue
                lines.append(LLMTextLine(text=t, bbox=_scale_bbox(entry.get('bbox'), image_size)))
        return lines

    # fallback: plain text, one line per physical line
    for row in text.splitlines():
        row = row.strip()
        if row:
            lines.append(LLMTextLine(text=row))
    return lines


class LLMOCRAdapter(ABC):
    DEFAULT_MODEL: str = ''

    def __init__(self,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_url: Optional[str] = None,
                 prompt: Optional[str] = None):
        self.model = model if model else self.DEFAULT_MODEL
        self.api_key = api_key
        self.api_url = api_url
        self.prompt = prompt

    @classmethod
    def is_available(cls) -> bool:
        """Whether this provider is usable on this server (required packages
        installed, API keys configured via environment variables, ...).
        Used to enable/disable the provider in the client."""
        return True

    @abstractmethod
    def transcribe(self, image: Image.Image) -> LLMPageTranscription:
        pass


class HuggingFaceVLMAdapter(LLMOCRAdapter):
    """Runs an open vision language model locally via HuggingFace transformers.

    Default model is Chandra 2 (datalab-to/chandra-ocr-2, 4B parameters); any
    other image-text-to-text model id (e.g. a Qwen-VL derivative) can be set
    via the ``llmModel`` parameter.
    """
    DEFAULT_MODEL = 'datalab-to/chandra-ocr-2'

    # cache loaded models across predictor instances, keyed by model id
    _CACHE: Dict[str, Tuple[object, object]] = {}

    @classmethod
    def is_available(cls) -> bool:
        import importlib.util
        if (importlib.util.find_spec('transformers') is None
                or importlib.util.find_spec('torch') is None):
            return False
        # running a multi-billion parameter VLM on CPU takes hours per page and
        # effectively hangs the task/tests, so require a GPU
        import torch
        return torch.cuda.is_available()

    def _load(self):
        if self.model not in self._CACHE:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "The '{}' text transcription requires a GPU; refusing to run on CPU.".format(self.model))
            from transformers import AutoProcessor, AutoModelForImageTextToText
            logger.info("Loading HuggingFace VLM '%s' (this may take a while)", self.model)
            processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
            model = AutoModelForImageTextToText.from_pretrained(
                self.model,
                dtype='auto',
                device_map='auto' if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            model.eval()
            self._CACHE[self.model] = (processor, model)
        return self._CACHE[self.model]

    def transcribe(self, image: Image.Image) -> LLMPageTranscription:
        import torch
        processor, model = self._load()
        prompt = build_prompt(image, self.prompt)
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        raw = processor.batch_decode(
            generated[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )[0]
        return LLMPageTranscription(lines=parse_llm_response(raw, image.size), raw_response=raw)


class GeminiAdapter(LLMOCRAdapter):
    """Google Gemini via the google-genai SDK.

    Enabled by setting the GEMINI_API_KEY (or GOOGLE_API_KEY) environment
    variable on the server; GEMINI_MODEL optionally overrides the default
    model. API keys are never accepted from the client.
    """
    DEFAULT_MODEL = 'gemini-2.5-flash'

    @classmethod
    def is_available(cls) -> bool:
        import importlib.util
        if not (os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')):
            return False
        return importlib.util.find_spec('google') is not None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not kwargs.get('model'):
            self.model = os.environ.get('GEMINI_MODEL') or self.DEFAULT_MODEL

    def transcribe(self, image: Image.Image) -> LLMPageTranscription:
        from google import genai
        api_key = self.api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("No API key for Gemini. Set the llmApiKey parameter or the GEMINI_API_KEY environment variable.")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=[build_prompt(image, self.prompt), image],
        )
        raw = response.text or ''
        return LLMPageTranscription(lines=parse_llm_response(raw, image.size), raw_response=raw)


class OpenAICompatibleAdapter(LLMOCRAdapter):
    """Any OpenAI compatible chat completions endpoint (OpenAI, OpenRouter,
    vLLM/LM Studio serving a local model, ...).

    Enabled by setting the OPENAI_API_KEY and/or OPENAI_API_URL environment
    variables on the server (a local endpoint such as vLLM needs no key);
    OPENAI_MODEL optionally overrides the default model. API keys are never
    accepted from the client.
    """
    DEFAULT_MODEL = 'gpt-4o'
    DEFAULT_API_URL = 'https://api.openai.com/v1'

    @classmethod
    def is_available(cls) -> bool:
        return bool(os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_URL'))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not kwargs.get('model'):
            self.model = os.environ.get('OPENAI_MODEL') or self.DEFAULT_MODEL

    def transcribe(self, image: Image.Image) -> LLMPageTranscription:
        import requests
        api_url = (self.api_url or os.environ.get('OPENAI_API_URL') or self.DEFAULT_API_URL).rstrip('/')
        api_key = self.api_key or os.environ.get('OPENAI_API_KEY', '')

        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=90)
        image_b64 = base64.b64encode(buffer.getvalue()).decode('ascii')

        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = 'Bearer ' + api_key
        body = {
            'model': self.model,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': build_prompt(image, self.prompt)},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + image_b64}},
                ],
            }],
            'max_tokens': 4096,
            'temperature': 0,
        }
        r = requests.post(api_url + '/chat/completions', headers=headers, json=body, timeout=600)
        r.raise_for_status()
        raw = r.json()['choices'][0]['message']['content'] or ''
        return LLMPageTranscription(lines=parse_llm_response(raw, image.size), raw_response=raw)


ADAPTER_REGISTRY: Dict[str, Type[LLMOCRAdapter]] = {
    'chandra': HuggingFaceVLMAdapter,
    'huggingface': HuggingFaceVLMAdapter,
    'gemini': GeminiAdapter,
    'openai': OpenAICompatibleAdapter,
}


def available_providers() -> Dict[str, bool]:
    """Availability of each registered provider on this server, used by the
    client to enable/disable the provider selection."""
    return {name: adapter.is_available() for name, adapter in ADAPTER_REGISTRY.items()}


def create_adapter(provider: Optional[str],
                   model: Optional[str] = None,
                   api_key: Optional[str] = None,
                   api_url: Optional[str] = None,
                   prompt: Optional[str] = None) -> LLMOCRAdapter:
    provider = (provider or 'chandra').lower()
    if provider not in ADAPTER_REGISTRY:
        raise ValueError("Unknown LLM provider '{}'. Available: {}".format(provider, ', '.join(sorted(ADAPTER_REGISTRY))))
    return ADAPTER_REGISTRY[provider](model=model, api_key=api_key, api_url=api_url, prompt=prompt)
