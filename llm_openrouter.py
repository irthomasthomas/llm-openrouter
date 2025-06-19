import click
import llm
import sys
from pathlib import Path
from pydantic import ConfigDict, Field, field_validator
from typing import Optional, Union, Dict, Any, List
import json
import time
import httpx

# Import correct base classes from default plugins
from llm.default_plugins.openai_models import Chat, AsyncChat

class _mixin:
    _saved_options: Optional[Dict[str, Any]] = None # Keep structure, but won't be used

    # Simplified __init__ needed by base classes if necessary
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_options = {}


    # Inherit from Chat.Options now, as per working pattern
    class Options(Chat.Options):
        model_config = ConfigDict(extra="allow") # Should allow unknown options

        # Caching options (for OpenRouter platform caching, but define)
        cache_mode: Optional[str] = Field(
            description="OpenRouter cache mode (e.g., semantic)",
            default=None
        )
        cache_max_age_seconds: Optional[int] = Field(
            description="OpenRouter cache lifetime in seconds",
            default=None
        )

        # Existing OpenRouter options
        online: Optional[bool] = Field(
            description="Use relevant search results from Exa",
            default=None,
        )
        provider: Optional[Union[dict, str]] = Field(
            description="JSON object to control provider routing",
            default=None,
        )

        @field_validator("provider")
        def validate_provider(cls, provider):
            if provider is None:
                return None
            if isinstance(provider, str):
                try:
                    return json.loads(provider)
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON in provider string")
            return provider

    # CORRECTED & REVISED build_kwargs logic (Access model_id via self.model_id)
    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)

        # Get the model ID using self.model_id (Corrected access)
        model_id = self.model_id

        # Check if it's an Anthropic model by prefix
        is_anthropic = model_id.startswith('openrouter/anthropic/')

        # Define the cache breakpoint marker
        CACHE_MARKER = "--- CACHE BREAKPOINT ---"

        extra_body = kwargs.get('extra_body', {})
        options = prompt.options # Get options from prompt (CLI/alias)

        # Add OpenRouter-specific parameters (like online, provider) to extra_body
        openrouter_params_in_extra_body = ['online', 'provider']
        for key in openrouter_params_in_extra_body:
            value = getattr(options, key, None)
            if value is not None:
                 if key == 'online':
                      extra_body['plugins'] = [{'id': 'web'}]
                 elif key == 'provider':
                      provider_val = value
                      if isinstance(provider_val, str):
                          try:
                              provider_val = json.loads(provider_val)
                          except json.JSONDecodeError:
                               print(f"Warning: Could not parse {key} JSON string from options: {provider_val}", file=sys.stderr)
                               provider_val = None
                      if provider_val is not None:
                           extra_body['provider'] = provider_val

        # Handle Anthropic native caching only if it's an Anthropic model
        # Modify the existing messages structure from prompt.messages
        if is_anthropic and isinstance(prompt.messages, list): # Ensure messages is a list
             original_messages = prompt.messages

             new_messages = []

             for message in original_messages:
                  new_content = []
                  # Check if message content is a list of content blocks
                  if isinstance(message.get('content'), list):
                      for content_block in message['content']:
                           # Process only text blocks
                           if content_block.get('type') == 'text' and isinstance(content_block.get('text'), str):
                                # Split this text block by the marker
                                text_segments = content_block['text'].split(CACHE_MARKER)

                                # Add segments to new_content, adding cache_control to segments after the first
                                for i, segment in enumerate(text_segments):
                                     if not segment.strip(): continue # Skip empty segments

                                     segment_block = {"type": "text", "text": segment.strip()}
                                     # Add cache_control to segments > 0 (those that followed a marker)
                                     if i > 0:
                                         segment_block["cache_control"] = {"type": "ephemeral"} # Using ephemeral as in example
                                     new_content.append(segment_block)

                           else:
                                # Keep non-text content blocks as they are (e.g., image_url)
                                new_content.append(content_block)
                  # If content is a single string
                  elif isinstance(message.get('content'), str):
                      # Split the single string content by marker
                      text_segments = message['content'].split(CACHE_MARKER)

                      # Add segments to new_content, adding cache_control to segments after the first
                      for i, segment in enumerate(text_segments):
                           if not segment.strip(): continue # Skip empty segments

                           segment_block = {"type": "text", "text": segment.strip()}
                           # Add cache_control to segments > 0 (those that followed a marker)
                           if i > 0:
                                segment_block["cache_control"] = {"type": "ephemeral"} # Using ephemeral as in example
                           new_content.append(segment_block)

                  # If content is neither list nor string, keep original content
                  else:
                       new_content = message.get('content')


                  # Add the message with potentially modified content to new_messages
                  new_message = message.copy()
                  new_message['content'] = new_content
                  new_messages.append(new_message)


             # Replace the original messages in kwargs with our new structure if Anthropic
             if new_messages: # Ensure our processing resulted in messages
                  kwargs['messages'] = new_messages
             # If original_messages was a list but our processing resulted in empty new_messages (e.g., all parts were empty)
             elif isinstance(prompt.messages, list):
                  kwargs['messages'] = new_messages


        # Update extra_body in kwargs
        if extra_body:
            kwargs['extra_body'] = extra_body
        else:
            kwargs.pop('extra_body', None) # Remove if empty


        # Keep debug prints
        print("--- Debug: build_kwargs: Model ID:", model_id, "Is Anthropic:", is_anthropic, file=sys.stderr)
        print("--- Debug: build_kwargs: prompt.options contents:", file=sys.stderr)
        try:
            print(options.model_dump_json(exclude_none=True), file=sys.stderr)
        except Exception as e:
            print(f"Could not dump prompt.options as JSON: {e}. Trying vars()", file=sys.stderr)
            print(vars(options), file=sys.stderr)
        print("--- End Debug: prompt.options ---", file=sys.stderr)
        print("--- Debug: build_kwargs: extra_body contents:", file=sys.stderr)
        print(json.dumps(extra_body), file=sys.stderr)
        print("--- Debug: build_kwargs: Final messages structure:", file=sys.stderr)
        print(json.dumps(kwargs.get('messages'), indent=2), file=sys.stderr) # Pretty print messages
        print("--- End Debug: Final messages ---", file=sys.stderr)

        return kwargs # Return the modified kwargs dictionary


# Define Chat and AsyncChat classes
class OpenRouterChat(_mixin, Chat):
    needs_key = "openrouter"
    key_env_var = "OPENROUTER_KEY"
    Options = _mixin.Options # Explicitly attach

    def __str__(self):
        return "OpenRouter: {}".format(self.model_id)

class OpenRouterAsyncChat(_mixin, AsyncChat):
    needs_key = "openrouter"
    key_env_var = "OPENROUTER_KEY"
    Options = _mixin.Options # Explicitly attach

    def __str__(self):
        return "OpenRouter: {}".format(self.model_id)


# Helper function (minimal) for register_models
def get_openrouter_models_minimal():
     return [
          {"id": "anthropic/claude-3-haiku", "name": "Claude 3 Haiku", "context_length": 200000, "supports_schema": False, "pricing": {}},
          {"id": "google/gemini-pro-1.5", "name": "Gemini 1.5 Pro", "context_length": 1000000, "supports_schema": True, "pricing": {}}
     ]

@llm.hookimpl
def register_models(register):
    key = llm.get_key("", "openrouter", "OPENROUTER_KEY")
    if not key:
        return

    for model_definition in get_openrouter_models_minimal():
        kwargs = dict(
            model_id="openrouter/{}".format(model_definition["id"]),
            model_name=model_definition["id"],
            vision=False, # Simplified
            supports_schema=model_definition["supports_schema"],
            api_base="https://openrouter.ai/api/v1",
            headers={"HTTP-Referer": "https://llm.datasette.io/", "X-Title": "LLM"},
        )

        chat_model = OpenRouterChat(**kwargs)
        async_chat_model = OpenRouterAsyncChat(**kwargs)
        register(
            chat_model,
            async_chat_model,
        )

# Exclude register_commands hook

