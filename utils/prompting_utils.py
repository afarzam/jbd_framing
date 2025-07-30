import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.model_utils import load_model

HF_token = "YOUR_HUGGING_FACE_TOKEN_HERE"

openai_API_key = \
"""
YOUR_OPENAI_API_KEY_HERE
"""
openai_API_key = openai_API_key.strip()



# --- HuggingFace standard wrapper ---

# utils/prompting_utils.py
import torch, json, re
from utils.model_utils import load_model

class HFSessionChat:
    """
    Minimal chat-wrapper that works for *any* HF chat model, whether or not
    its template supports a `system` role.  Public API is unchanged:
        * set_instructions(text)
        * prompt(user_prompt, ...)
    """

    def __init__(self, model=None, tokenizer=None, *,
                 model_name=None, device=None,
                 load_in_8bit=False, load_in_4bit=False):

        if model is None:
            assert model_name, "Need either a model obj or model_name"
            model, tokenizer = load_model(model_name,
                                          device=device,
                                          load_in_8bit=load_in_8bit,
                                          load_in_4bit=load_in_4bit)
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = self.model.device
        self._system   = ""          # raw instruction string
        self._history  = []          # list[dict(role, content)]

        # Does the template recognise a system role?
        tpl = getattr(self.tokenizer, "chat_template", "")
        self._has_system_slot = ("role == 'system'" in tpl) or ('"system"' in tpl)

    # ---------- public helpers ----------
    def set_instructions(self, text: str):
        self._system = text or ""

    def reset_session(self):
        self._history.clear()

    def erase_history(self):
        self._history.clear()
        self._system = ""

    def get_history(self):
        return list(self._history)

    # ---------- core call ----------
    @torch.inference_mode()
    def prompt(self, user_prompt, *, max_new_tokens=256,
               continue_session=True, **generate_kwargs):

        # 1) build message list
        messages = []
        if self._has_system_slot and self._system:
            messages.append({"role": "system", "content": self._system})

        # prepend instructions if no system slot
        if not self._has_system_slot and self._system:
            user_prompt = f"{self._system}\n\n{user_prompt}"

        if continue_session:
            messages.extend(self._history)

        messages.append({"role": "user", "content": user_prompt})

        # 2) encode
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        # 3) generate with proper parameters
        # Filter out invalid parameters
        valid_params = {}
        for key, value in generate_kwargs.items():
            if key in ['do_sample', 'temperature', 'top_p', 'top_k', 'repetition_penalty', 
                      'pad_token_id', 'eos_token_id', 'max_new_tokens', 'num_return_sequences']:
                valid_params[key] = value
        
        # Set default parameters if not provided
        if 'do_sample' not in valid_params:
            valid_params['do_sample'] = True
        if 'temperature' not in valid_params:
            valid_params['temperature'] = 0.7
        if 'top_p' not in valid_params:
            valid_params['top_p'] = 0.9
        if 'repetition_penalty' not in valid_params:
            valid_params['repetition_penalty'] = 1.1

        out_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            **valid_params
        )
        reply = self.tokenizer.decode(
            out_ids[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()

        # 4) save history
        if continue_session:
            self._history.extend([
                {"role": "user",      "content": user_prompt},
                {"role": "assistant", "content": reply},
            ])

        return {"response": reply, "history": self.get_history()}



# class HFSessionChat:
#     def __init__(self, model=None, tokenizer=None, model_name=None, device=None, load_in_8bit=False, load_in_4bit=False):

#         if model is None:
#             assert model_name, "Need either a model obj or model_name"
#             model, tokenizer = load_model(model_name,
#                                           device=device,
#                                           load_in_8bit=load_in_8bit,
#                                           load_in_4bit=load_in_4bit)
#         self.model      = model
#         self.tokenizer  = tokenizer
#         self.device     = self.model.device     # after load_model this is set
#         self._system    = ""                    # default empty system prompt
#         self._history   = []                    # list[dict(role, content)]

#     # ---------- public helpers ----------
#     def set_instructions(self, text: str):
#         self._system = text or ""

#     def reset_session(self):
#         self._history.clear()

#     def erase_history(self):
#         self._history.clear()
#         self._system = ""

#     def get_history(self):
#         return list(self._history)

#     # ---------- core call ----------
#     @torch.inference_mode()
#     def prompt(self, user_prompt, max_new_tokens=1024, continue_session=True, **generate_kwargs):

#         # ------------------ find out if template knows "system" -------------
#         template_str = getattr(self.tokenizer, "chat_template", "")
#         has_system = "role == 'system'" in template_str or '"system"' in template_str

#         # ------------------  build messages  --------------------------------
#         messages = []
#         if has_system and self._system:
#             messages.append({"role": "system", "content": self._system})

#         # prepend instructions to first user turn if no system slot
#         full_user_msg = (
#             ("" if has_system else (self._system + "\n\n")) + user_prompt
#         )
#         if continue_session:
#             messages.extend(self._history)
#         messages.append({"role": "user", "content": full_user_msg})

#         # ------------------  encode & generate  -----------------------------
#         input_ids = self.tokenizer.tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             return_tensors="pt"
#         ).to(device)

#         output_ids = self.model.generate(
#             input_ids,
#             max_new_tokens=max_new_tokens,
#             eos_token_id=self.tokenizer.eos_token_id,
#             **generate_kwargs
#         )

#         reply = self.tokenizer.decode(
#             output_ids[0][input_ids.shape[-1]:],
#             skip_special_tokens=True
#         ).strip()

#         # ------------------  save history  ----------------------------------
#         if continue_session:
#             self._history.append({"role": "user",      "content": full_user_msg})
#             self._history.append({"role": "assistant", "content": reply})

#         return {"response": reply, "history": list(self._history)}





# --- HuggingFace Prompting Utilities ---

class HFSession:
    def __init__(self, model=None, tokenizer=None, model_name=None, device=None, load_in_8bit=False, load_in_4bit=False):
        if model:
            assert tokenizer is not None, "Tokenizer must be provided if model is provided"
            self.model = model
            self.tokenizer = tokenizer
            self.device = self.model.device
        else:
            assert model_name is not None, "Model name must be provided if model is not provided"
            self.model, self.tokenizer = load_model(
                model_name, device=device, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
            )
            self.device = self.model.device
        self._instructions = None
        self._history = []

    def set_instructions(self, instructions):
        """Set instructions for the session (prepended to the first prompt)."""
        self._instructions = instructions

    def reset_session(self):
        """Reset session history, keeping instructions if set."""
        self._history = []

    def erase_history(self):
        """Erase all history and instructions."""
        self._history = []
        self._instructions = None

    def get_history(self):
        """Return the current session history as a list of dicts."""
        return list(self._history)

    def prompt(self, prompt_text, max_new_tokens=1024, continue_session=True, **generate_kwargs):
        """
        Prompt the HuggingFace model, optionally continuing a session.
        Returns: dict with 'response', 'history'
        """
        # Compose prompt from instructions and history
        full_prompt = ""
        if self._instructions:
            full_prompt += self._instructions + "\n"
        if continue_session and self._history:
            for turn in self._history:
                full_prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        full_prompt += f"User: {prompt_text}\nAssistant:"

        inputs = self.tokenizer(full_prompt, return_tensors='pt').to(self.device)
        
        # Filter out invalid parameters and set defaults
        valid_params = {}
        for key, value in generate_kwargs.items():
            if key in ['do_sample', 'temperature', 'top_p', 'top_k', 'repetition_penalty', 
                      'pad_token_id', 'eos_token_id', 'max_new_tokens', 'num_return_sequences']:
                valid_params[key] = value
        
        # Set default parameters if not provided
        if 'do_sample' not in valid_params:
            valid_params['do_sample'] = True
        if 'temperature' not in valid_params:
            valid_params['temperature'] = 0.7
        if 'top_p' not in valid_params:
            valid_params['top_p'] = 0.9
        if 'repetition_penalty' not in valid_params:
            valid_params['repetition_penalty'] = 1.1
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            **valid_params
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Only keep the new assistant response (after "Assistant:")
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        if continue_session:
            self._history.append({"user": prompt_text, "assistant": response})
        return {"response": response, "history": self.get_history()}



class HFSession_pre:
    """
    Session compatible with pretrained (non instruction-tuned) HF models.
    It builds a plain-text prompt by concatenating optional instructions and conversation history.
    API is similar to:
        * set_instructions(text)
        * reset_session()
        * erase_history()
        * get_history()
        * prompt(prompt_text, ...)
    """
    def __init__(self, model=None, tokenizer=None, model_name=None, device=None, load_in_8bit=False, load_in_4bit=False):
        if model:
            assert tokenizer is not None, "Tokenizer must be provided if model is provided"
            self.model = model
            self.tokenizer = tokenizer
            self.device = self.model.device
        else:
            assert model_name is not None, "Model name must be provided if model is not provided"
            from utils.model_utils import load_model
            self.model, self.tokenizer = load_model(
                model_name, device=device, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
            )
            self.device = self.model.device
        self._instructions = None
        self._history = []

    def set_instructions(self, instructions):
        """Set instructions (prepended to the prompt)."""
        self._instructions = instructions

    def reset_session(self):
        """Reset session history (keeps instructions, if any)."""
        self._history = []

    def erase_history(self):
        """Erase all history and instructions."""
        self._history = []
        self._instructions = None

    def get_history(self):
        """Return a copy of session history."""
        return list(self._history)

    def prompt(self, prompt_text, max_new_tokens=1024, continue_session=True, **generate_kwargs):
        """
        Prompt the model with a plain-text prompt combining instructions and conversation history.
        Returns a dict with keys: 'response', 'history'
        """
        # Build the full prompt
        full_prompt = ""
        if self._instructions:
            full_prompt += self._instructions + "\n\n"
        if continue_session and self._history:
            for turn in self._history:
                full_prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        full_prompt += f"User: {prompt_text}\nAssistant:"

        # Encode the prompt
        inputs = self.tokenizer(full_prompt, return_tensors='pt').to(self.device)
        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            **generate_kwargs
        )
        # Decode the output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Check for the custom marker for new scenario output
        marker = "\n new scenario is:\n"
        if marker in response:
            response = response.split(marker)[-1].strip()
        elif "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        else:
            response = response.strip()
        # Save history if continuing session
        if continue_session:
            self._history.append({
                "user": prompt_text,
                "assistant": response
            })
        return {"response": response, "history": self.get_history()}


# --- OpenAI Prompting Utilities ---

from openai import OpenAI

client = OpenAI(api_key=openai_API_key)

class OpenAISession:
    def __init__(self, model_name="gpt-4.1"):
        self.model_name = model_name
        self._instructions = None
        self._history = []

    def set_instructions(self, instructions):
        """Set system instructions for the session."""
        self._instructions = instructions

    def reset_session(self):
        """Reset session history, keeping instructions if set."""
        self._history = []

    def erase_history(self):
        """Erase all history and instructions."""
        self._history = []
        self._instructions = None

    def get_history(self):
        """Return the current session history as a list of dicts."""
        return list(self._history)

    def prompt(self, prompt_text, max_new_tokens=None, continue_session=True, **kwargs):
        """
        Prompt the OpenAI model, optionally continuing a session.
        Returns: dict with 'response', 'history'
        """
        messages = []
        if self._instructions:
            messages.append({"role": "developer", "content": self._instructions})
        if continue_session and self._history:
            for turn in self._history:
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": prompt_text})
        response_obj = client.chat.completions.create(model=self.model_name,
        messages=messages,
        **kwargs)
        reply = response_obj.choices[0].message.content
        if continue_session:
            self._history.append({"user": prompt_text, "assistant": reply})
        return {"response": reply, "history": self.get_history()}



# --- Example Usage ---

# hf_session = HFSession(model_name="meta-llama/Llama-2-7b-chat-hf")
# hf_session.set_instructions("You are a helpful assistant.")
# out = hf_session.prompt("Hello, how are you?")
# print(out["response"])

# openai_session = OpenAISession(api_key="sk-...")
# openai_session.set_instructions("You are a helpful assistant.")
# out = openai_session.prompt("Hello, how are you?")
# print(out["response"])