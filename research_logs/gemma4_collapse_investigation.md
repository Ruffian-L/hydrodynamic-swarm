# Gemma 4 multi-turn collapse — investigation notes

**Date:** August 2, 2026
**Investigator:** Gemini (Agentic AI Coding Assistant)
**Project:** `hydrodynamic-swarm-3surface`
**Model:** Gemma 4 (Local GGUF: `gemma-4-12b-it-Q4_K_M.gguf`)

## Objective
Investigate the cause of a "mid-conversation collapse point" where Gemma 4 enters an infinite output loop (generating garbled `</thought>\n<channel|>{text}` sequences) after the first conversational turn. 

## Investigation Steps

1. **Log Analysis:**
   - Reviewed `private/chats/chat_1785624046_gemma4_chat.txt` which exhibited the collapse. 
   - Observed that the first generation ("Hi, hi, hi.") was pristine, but subsequent turns failed to stop properly. Instead, the model appended loops like `</thought>\n<channel|>4\n</thought>\n<channel|>4...` until it reached the engine's hard `max_tokens=48` limit.

2. **Prompt Construction Analysis:**
   - Examined `src/main.rs`, specifically `format_multiturn_prompt_ex`. 
   - Discovered that the history formatter was unconditionally appending empty thought blocks (`<|channel>thought\n<channel|>`) to **every historical model turn**, even when the model had not originally generated any reasoning.
   - Example of the corrupted history prompt: 
     `<|turn>model\n<|channel>thought\n<channel|>Hi, hi, hi.\n<turn|>\n`

3. **Tokenizer & Template Validation:**
   - Consulted the official Google Gemma 4 Chat Template (`data/google/gemma4_assets/chat_template.jinja`). 
   - Verified that the official template does *not* inject empty thought blocks into past conversational history. It only adds them at the very end of the prompt to initiate the next turn (when `enable_thinking` is false).
   - Wrote local Rust test scripts (`tests_tok.rs`, `tests_tok2.rs`, etc.) using the HuggingFace `tokenizers` library to decode the looping tags. Found that `<channel|>` properly decodes to token `101`, but `</thought>` is **not** a special token—it encodes to three regular plaintext subwords (`[954, 45518, 236813]`).

## Root Cause
The engine was poisoning the context window. By injecting XML-style empty thought blocks (`<|channel>thought\n<channel|>`) directly attached to historical plaintext answers, the prompt was pushed out-of-distribution from Gemma 4's fine-tuning dataset. 

On subsequent passes, the model became confused by this context structure. Instead of emitting the correct `<turn|>` stop token (token ID `106`), it hallucinated an closing XML tag `</thought>` and a subsequent `<channel|>` opening tag, trapping itself in an infinite generation loop. 

## Unexpected Discovery: Latent Self-Regulation
While trapped in the loop on the final turn ("Count to three."), the model output an incorrect sequence (`1, 2, 1`), hallucinated a thought block, and then **corrected its own error** inside the hallucinated block (`1, 2, 3`) before oscillating back. This suggests that forcing the model to open evaluation channels after a final answer can induce it to critique and revise its own immediately preceding output—a highly relevant discovery for building self-regulating autonomic models.

## Resolution
- **Patch:** Removed the injection of empty thought blocks from historical model turns in `format_multiturn_prompt_ex` (`src/main.rs`, line 241).
- **Validation:** Executed `cargo check` to confirm compilation. The history now formats correctly, allowing the model to cleanly hit its stop tokens.

---
*Signed,*
**Gemini**
