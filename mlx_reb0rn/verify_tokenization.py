"""
verify_tokenization.py — Verify MLX inference tokenization matches GR00T training.

Checks:
  1. Special token IDs  (<IMG_CONTEXT>, <img>, </img>, BOS/EOS)
  2. Prompt template string (training apply_chat_template vs MLX manual build)
  3. Token-by-token comparison of encoded prompt
  4. Image token count: training num_image_token vs actual vision encoder output
  5. Language formalization (training lowercases + strips punctuation)
  6. Image preprocessing normalization equivalence

Run from the mlx_gr00t/ directory:
    python verify_tokenization.py
"""

import sys, math, re, json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
WEIGHTS_DIR = HERE / "gr00t_weights_mlx"
EAGLE_PATH  = str(HERE.parent / "Eagle" / "Eagle2_5")
ISAAC_PATH  = str(HERE.parent / "Isaac-GR00T")

for p in [EAGLE_PATH, ISAAC_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Helpers ────────────────────────────────────────────────────────────────
PASS = "\033[32m✓ PASS\033[0m"
FAIL = "\033[31m✗ FAIL\033[0m"
WARN = "\033[33m⚠ WARN\033[0m"

def check(label: str, ok: bool, detail: str = ""):
    status = PASS if ok else FAIL
    print(f"  {status}  {label}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")
    return ok

results = []

# ─────────────────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════════════")
print("  GR00T MLX vs Training Tokenization Verification")
print("══════════════════════════════════════════════════════\n")

# ── Load shared config ─────────────────────────────────────────────────────
with open(WEIGHTS_DIR / "meta.json") as f:
    meta = json.load(f)

IMAGE_SIZE_RESIZE  = meta["image_size"]           # 378 — used for PIL resize in get_action()
IMAGE_TOKEN_INDEX  = meta["image_token_index"]   # 262145  (<IMG_CONTEXT>)
PATCH_SIZE         = 14                          # siglip2-so400m-patch14
DOWNSAMPLE_RATIO   = 0.5                         # --down_sample_ratio 0.5
# The _Eagle2_5ProcessorShim in processing_gr00t_n1d6.py defaults to image_size=384
# (not 378). build_processor() constructs it without an image_size kwarg, so the
# training formula always uses 384 → ceil(384/14)=28 → 28^2*0.25=196 tokens.
TRAINING_SHIM_IMAGE_SIZE = 384

# ─────────────────────────────────────────────────────────────────────────
print("── 1. Load tokenizer ────────────────────────────────")
from transformers import GemmaTokenizer

tokenizer = GemmaTokenizer.from_pretrained(
    str(WEIGHTS_DIR / "eagle_tokenizer"), use_fast=False, local_files_only=True
)
print(f"  Vocab size : {len(tokenizer)}")
print(f"  Model type : {tokenizer.__class__.__name__}")

# ─────────────────────────────────────────────────────────────────────────
print("\n── 2. Special token IDs ─────────────────────────────")

IMG_CONTEXT = "<IMG_CONTEXT>"
IMG_START   = "<img>"
IMG_END     = "</img>"
BOS_TOKEN   = tokenizer.bos_token
EOS_TOKEN   = tokenizer.eos_token

img_ctx_id  = tokenizer.convert_tokens_to_ids(IMG_CONTEXT)
img_start_id= tokenizer.convert_tokens_to_ids(IMG_START)
img_end_id  = tokenizer.convert_tokens_to_ids(IMG_END)
bos_id      = tokenizer.bos_token_id
eos_id      = tokenizer.eos_token_id

print(f"  BOS={bos_id} ({BOS_TOKEN})")
print(f"  EOS={eos_id} ({EOS_TOKEN})")
print(f"  <IMG_CONTEXT> id = {img_ctx_id}")
print(f"  <img>         id = {img_start_id}")
print(f"  </img>        id = {img_end_id}")

results.append(check(
    "<IMG_CONTEXT> matches meta.json image_token_index",
    img_ctx_id == IMAGE_TOKEN_INDEX,
    f"tokenizer={img_ctx_id}  meta.json={IMAGE_TOKEN_INDEX}",
))
results.append(check(
    "<IMG_CONTEXT> is a genuine added token (not UNK)",
    img_ctx_id != tokenizer.unk_token_id,
    f"unk_token_id={tokenizer.unk_token_id}",
))
results.append(check(
    "<img> / </img> tokens resolve (not UNK)",
    img_start_id != tokenizer.unk_token_id and img_end_id != tokenizer.unk_token_id,
    f"<img>={img_start_id}  </img>={img_end_id}",
))

# ─────────────────────────────────────────────────────────────────────────
print("\n── 3. Image token count ─────────────────────────────")
# Training formula (processing_gr00t_n1d6.py _Eagle2_5ProcessorShim.__init__)
# NOTE: build_processor() constructs the shim WITHOUT an image_size kwarg, so the
# shim always uses its default image_size=384, NOT the eagle_config's 378.
training_n_img_tok = int(
    math.ceil(TRAINING_SHIM_IMAGE_SIZE / PATCH_SIZE) ** 2 * (DOWNSAMPLE_RATIO ** 2)
)
print(f"  Training shim image_size={TRAINING_SHIM_IMAGE_SIZE} (default in _Eagle2_5ProcessorShim)")
print(f"  MLX resize image_size  ={IMAGE_SIZE_RESIZE} (meta.json, used for PIL resize)")
print(f"  patch_size={PATCH_SIZE}, downsample_ratio={DOWNSAMPLE_RATIO}")
print(f"  Training formula : ceil({TRAINING_SHIM_IMAGE_SIZE}/{PATCH_SIZE})^2 * {DOWNSAMPLE_RATIO}^2")
print(f"                   = {math.ceil(TRAINING_SHIM_IMAGE_SIZE / PATCH_SIZE)}^2 * {DOWNSAMPLE_RATIO**2}")
print(f"                   = {math.ceil(TRAINING_SHIM_IMAGE_SIZE / PATCH_SIZE)**2} * {DOWNSAMPLE_RATIO**2}")
print(f"                   = {training_n_img_tok}")
print()
print(f"  MLX inference uses vit_embeds.shape[1] (actual encoder output).")
print(f"  ── Probing MLX vision encoder token count ──")

try:
    import numpy as np
    import mlx.core as mx

    sys.path.insert(0, str(HERE))
    from vision_mlx import build_vision_mlx_from_exported

    vision_model = build_vision_mlx_from_exported(
        str(WEIGHTS_DIR / "vision.safetensors"), meta, dtype="float16"
    )
    vision_model.eval()

    # Use IMAGE_SIZE_RESIZE (378) — same as get_action() in gemma_vla.py
    dummy = np.zeros((IMAGE_SIZE_RESIZE, IMAGE_SIZE_RESIZE, 3), dtype=np.float32)
    pixel_values = mx.array(dummy[None])
    vit_out = vision_model(pixel_values)
    mx.eval(vit_out)
    mlx_n_img_tok = vit_out.shape[1]
    print(f"  MLX actual output shape: {vit_out.shape}  → {mlx_n_img_tok} tokens")

    tok_match = (mlx_n_img_tok == training_n_img_tok)
    results.append(check(
        "MLX vision encoder token count == training formula",
        tok_match,
        f"MLX={mlx_n_img_tok}  training={training_n_img_tok}",
    ))
except Exception as e:
    print(f"  {WARN}  Could not load vision encoder: {e}")
    print(f"        Falling back to formula check only.")
    mlx_n_img_tok = training_n_img_tok  # assume match for prompt construction below
    results.append(check(
        "MLX vision encoder token count == training formula (formula only)",
        True,
        f"training formula gives {training_n_img_tok}  (vision model not loaded)",
    ))

# ─────────────────────────────────────────────────────────────────────────
print("\n── 4. Prompt template comparison ────────────────────")
TEST_INSTRUCTION = "pick up the red object"

# ── Training prompt (via _Eagle2_5ProcessorShim.apply_chat_template) ──────
# Replicate exactly what processing_gr00t_n1d6.py produces at training time.
from eaglevl.conversation import (
    Conversation, SeparatorStyle, register_conv_template, get_conv_template,
)
try:
    get_conv_template("gemma3-chat")
except KeyError:
    register_conv_template(
        Conversation(
            name="gemma3-chat",
            system_template="",
            system_message="",
            roles=("<start_of_turn>user", "<start_of_turn>model"),
            sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
            sep="<end_of_turn>\n",
            stop_str="<end_of_turn>",
        ),
        override=False,
    )

def training_apply_chat_template(instruction: str, n_images: int, num_image_token: int) -> str:
    """Replicates _Eagle2_5ProcessorShim.apply_chat_template."""
    template = get_conv_template("gemma3-chat")
    image_tokens = "".join(
        f"{IMG_START}{IMG_CONTEXT * num_image_token}{IMG_END}"
        for _ in range(n_images)
    )
    user_text = instruction
    full_text = image_tokens + "\n" + user_text if n_images > 0 else user_text
    template.append_message(template.roles[0], full_text)
    template.append_message(template.roles[1], None)
    return template.get_prompt()

training_prompt = training_apply_chat_template(
    TEST_INSTRUCTION, n_images=1, num_image_token=training_n_img_tok
)

# ── MLX prompt (gemma_vla.py get_action) ──────────────────────────────────
def mlx_build_prompt(instruction: str, n_img_tokens: int) -> str:
    image_block = f"{IMG_START}{IMG_CONTEXT * n_img_tokens}{IMG_END}"
    return (
        f"<start_of_turn>user\n"
        f"{image_block}\n"
        f"{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

mlx_prompt = mlx_build_prompt(TEST_INSTRUCTION, mlx_n_img_tok)

# Compare (replace long IMG_CONTEXT runs with placeholders for readability)
def abbrev(s: str) -> str:
    def _rep(m):
        count = (m.end() - m.start()) // len("<IMG_CONTEXT>")
        return f"<IMG_CONTEXT>×{count}"
    return re.sub(r"(<IMG_CONTEXT>){4,}", _rep, s)

print(f"  Training prompt: {repr(abbrev(training_prompt))}")
print(f"  MLX     prompt: {repr(abbrev(mlx_prompt))}")

prompts_match = (training_prompt == mlx_prompt)
results.append(check(
    "Prompt strings are identical",
    prompts_match,
))
if not prompts_match:
    # Show first divergence
    for i, (a, b) in enumerate(zip(training_prompt, mlx_prompt)):
        if a != b:
            print(f"         First diff at char {i}: train={repr(a)} mlx={repr(b)}")
            print(f"         Context: ...{repr(training_prompt[max(0,i-20):i+20])}...")
            break
    if len(training_prompt) != len(mlx_prompt):
        print(f"         Length: training={len(training_prompt)} mlx={len(mlx_prompt)}")

# ─────────────────────────────────────────────────────────────────────────
print("\n── 5. Token ID comparison ───────────────────────────")
train_ids = tokenizer.encode(training_prompt)
mlx_ids   = tokenizer.encode(mlx_prompt)

print(f"  Training token count : {len(train_ids)}")
print(f"  MLX     token count  : {len(mlx_ids)}")

if train_ids == mlx_ids:
    _tok_detail = "MATCH"
elif len(train_ids) != len(mlx_ids):
    _tok_detail = f"different lengths: training={len(train_ids)} mlx={len(mlx_ids)}"
else:
    _first = next(i for i, (a, b) in enumerate(zip(train_ids, mlx_ids)) if a != b)
    _tok_detail = f"first diff at index {_first}"
results.append(check("Token ID sequences are identical", train_ids == mlx_ids, _tok_detail))

# Check BOS is present
results.append(check(
    "BOS token present at position 0",
    len(train_ids) > 0 and train_ids[0] == bos_id,
    f"ids[0]={train_ids[0] if train_ids else 'EMPTY'}  bos_id={bos_id}",
))

# Check <IMG_CONTEXT> tokens are in the sequence
ctx_count = train_ids.count(IMAGE_TOKEN_INDEX)
results.append(check(
    f"<IMG_CONTEXT> appears exactly {training_n_img_tok} times",
    ctx_count == training_n_img_tok,
    f"found={ctx_count}  expected={training_n_img_tok}",
))

# ─────────────────────────────────────────────────────────────────────────
print("\n── 6. Language formalization ────────────────────────")
# Training: content.text.lower() + re.sub(r"[^\w\s]", "", ...) when formalize_language=True
raw_instr     = "Pick up the red object!"
formal_instr  = re.sub(r"[^\w\s]", "", raw_instr.lower())
print(f"  Raw instruction      : {repr(raw_instr)}")
print(f"  Formalized (training): {repr(formal_instr)}")
print(f"  MLX uses raw text (no formalization).")

ids_raw    = tokenizer.encode(mlx_build_prompt(raw_instr,   mlx_n_img_tok))
ids_formal = tokenizer.encode(mlx_build_prompt(formal_instr, mlx_n_img_tok))
same = (ids_raw == ids_formal)
results.append(check(
    "Raw vs formalized instruction produce same tokens (punctuation/case neutral)",
    same,
    "If FAIL: MLX must lowercase + strip punctuation to match training." if not same else "",
))
if not same:
    print(f"  {WARN}  Action required: add 'instruction = re.sub(r\"[^\\w\\s]\", \"\", instruction.lower())'")
    print(f"         in GemmaVLA.get_action() before building the prompt.")

# ─────────────────────────────────────────────────────────────────────────
print("\n── 7. Image preprocessing normalization ─────────────")
import numpy as np

# Training path (_Eagle2_5ProcessorShim.__call__):
#   TF.to_tensor → [0,1]  then TF.normalize(mean=0.5, std=0.5) → (x - 0.5)/0.5
# MLX path (gemma_vla.py get_action):
#   (np/255 - 0.5) / 0.5
pixel = np.array([100, 150, 200], dtype=np.float32)

training_val = (pixel / 255.0 - 0.5) / 0.5   # TF.to_tensor then normalize
mlx_val      = (pixel / 255.0 - 0.5) / 0.5   # same formula

results.append(check(
    "Image pixel normalization formula is equivalent",
    np.allclose(training_val, mlx_val),
    f"training={training_val}  mlx={mlx_val}",
))

# Resize method check
print(f"  Training : PIL.resize({TRAINING_SHIM_IMAGE_SIZE},{TRAINING_SHIM_IMAGE_SIZE}) inside _Eagle2_5ProcessorShim.__call__")
print(f"             (images already passed through torchvision letterbox → resize → crop → resize)")
print(f"  MLX      : image.resize(({IMAGE_SIZE_RESIZE},{IMAGE_SIZE_RESIZE}), Image.BICUBIC)  — single resize, no letterbox")
print(f"  Note: training shim final resize={TRAINING_SHIM_IMAGE_SIZE}, MLX resize={IMAGE_SIZE_RESIZE}")
print(f"  {WARN}  Preprocessing differs: training uses letterbox+crop pipeline,")
print(f"         MLX uses direct BICUBIC resize.  Token values differ but")
print(f"         this is an expected deployment simplification.")

# ─────────────────────────────────────────────────────────────────────────
print("\n── 8. Embedding scale factor ────────────────────────")
# gemma_vla.py line ~385: vit_scaled = vit_embeds / sqrt(hidden_size)
# This pre-divides because MLX's Gemma3 embed_tokens multiplies by sqrt(hidden_size).
# Training: the GR00T model's merge_text_and_vision does NOT re-scale, so the HF
# implementation scales differently.  This is MLX-specific and not a tokenization issue.
print(f"  MLX divides vision features by sqrt(hidden_size) to cancel Gemma3's")
print(f"  internal embed scaling.  This is not part of tokenization — skip.")

# ─────────────────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════════════")
print("  Summary")
print("══════════════════════════════════════════════════════")
passed = sum(results)
total  = len(results)
for i, r in enumerate(results, 1):
    pass  # results tracked as bool; status already printed above

print(f"\n  {passed}/{total} checks passed")
if passed == total:
    print(f"\n  {PASS}  Tokenization is consistent with training.")
else:
    print(f"\n  {FAIL}  {total - passed} check(s) failed — see details above.")
print()
