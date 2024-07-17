from typing import Any, Dict, Optional
import json

# Import LlamaIndex dependencies
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.base.llms.generic_utils import (
    get_from_param_or_env,
)
from llama_index.core.llms.callbacks import llm_completion_callback, CallbackManager
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.bridge.pydantic import PrivateAttr, Field

from cleanlab_studio import Studio

DEFAULT_CONTEXT_WINDOW = 16385
DEFAULT_MAX_TOKENS = 512
DEFAULT_MODEL = "gpt-3.5-turbo-16k"


class CleanlabTLM(CustomLLM):
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        description="The maximum number of tokens to generate in TLM response.",
    )
    model: str = Field(
        default="gpt-3.5-turbo-16k", description="The base model to use."
    )
    quality_preset: str = Field(
        default="medium", description="Pre-defined configuration to use for TLM."
    )

    _client: Any = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        quality_preset: Optional[str] = "medium",
        options: Optional[Dict] = None,
        callback_manager: Optional[CallbackManager] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            additional_kwargs=additional_kwargs or {},
            callback_manager=callback_manager,
        )

        self.quality_preset = quality_preset
        use_options = options is not None
        # Check for user overrides in options dict
        if use_options:
            if options.get("model") is not None:
                self.model = options.get("model")
                if self.model == "gpt-4":
                    self.context_window = 8192
                elif self.model == "gpt-3.5-turbo-16k":
                    self.context_window = 16385
                else:
                    # ValueError is raised by Studio object for non-supported models
                    # Set context_window to dummy (default) value
                    self.context_window = DEFAULT_CONTEXT_WINDOW
            else:
                self.context_window = DEFAULT_CONTEXT_WINDOW

            if options.get("max_tokens") is not None:
                self.max_tokens = options.get("max_tokens")
            else:
                self.max_tokens = DEFAULT_MAX_TOKENS
        else:
            self.model = DEFAULT_MODEL
            self.context_window = DEFAULT_CONTEXT_WINDOW
            self.max_tokens = DEFAULT_MAX_TOKENS

        api_key = get_from_param_or_env("api_key", api_key, "CLEANLAB_API_KEY")

        studio = Studio(api_key=api_key)
        self._client = studio.TLM(
            quality_preset=self.quality_preset, options=options if use_options else None
        )

    @classmethod
    def class_name(cls) -> str:
        return "CleanlabTLM"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # Prompt TLM for a response and trustworthiness score
        response: Dict[str, str] = self._client.prompt(prompt)

        return CompletionResponse(
            text=response["response"],
            additional_kwargs={
                "trustworthiness_score": response["trustworthiness_score"]
            },
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # Prompt TLM for a response and trustworthiness score
        response = self._client.prompt(prompt)
        output = json.dumps(response)

        # TODO: figure how to stream additional_kwargs. workaround: dump `trustworthiness_score` as str
        # Stream the output
        output_str = ""
        for token in output:
            output_str += token
            yield CompletionResponse(text=output_str, delta=token)
