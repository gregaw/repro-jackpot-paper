"""Stage: Maximal Lottery policy via Online IPO (Calandriello et al. 2024,
arXiv:2403.08635), plus the offline-IPO control.

Online IPO: each step samples a batch of completions from the current policy,
labels every pair (i, j) with the empirical preference P(colour_i > colour_j),
and regresses the log-ratio difference onto it:

    h_ij = [log pi(y_i) - log ref(y_i)] - [log pi(y_j) - log ref(y_j)]
    loss = mean_{i<j} (h_ij - (P(c_i > c_j) - 1/2) / tau)^2

Its minimiser is the Nash equilibrium of the tau-regularised preference game
(Prop. 4.1), so the *last iterate* is the output policy; a mixture of iterates
is still accumulated, for comparison with SPO only. The reference is the same
model with the LoRA adapter switched off. Plain PyTorch: no PPO, no value head,
no forced-colour substitution (that would make the pairs partly off-policy).

Offline IPO (method "ipo_offline") is the same loss on the data_gen triplets,
`prompt + " chosen."` against `prompt + " rejected."` with target 1/(2 tau):
standard IPO on a fixed dataset. The 0086 tabular simulation predicts it picks
the Borda winner and flips on IIA, like RLHF.
"""

import random

from pipeline import manifest
from pipeline.config import prompt_for, train_params
from pipeline.costs import stage_timer
from pipeline.models import build_causal_lm, lora_config
from pipeline.populations import CONFIGS, empirical_preference
from pipeline.stages import train_common as tc
from pipeline.storage import Storage


def pair_preference(a, b, pref) -> float:
    """P(a > b) with SPO's rule for unparsed completions (None): an unnamed
    colour loses to any named one, and two unnamed ones tie."""
    if a is None and b is None:
        return 0.5
    if a is None:
        return 0.0
    if b is None:
        return 1.0
    return pref(a, b)


def pair_targets(parsed: list, pref, tau: float):
    """k x k matrix of IPO regression targets (P(c_i > c_j) - 1/2) / tau."""
    import torch

    k = len(parsed)
    t = torch.zeros(k, k)
    for i in range(k):
        for j in range(k):
            if i != j:
                t[i, j] = (pair_preference(parsed[i], parsed[j], pref) - 0.5) / tau
    return t


def online_ipo_loss(log_ratio, targets, weights=None):
    """Mean squared error of h_ij = d_i - d_j against targets[i, j] over the
    pairs i < j of a batch (`weights`: optional k x k pair weights instead of a
    uniform mean, used by the exact-expectation test)."""
    import torch

    h = log_ratio[:, None] - log_ratio[None, :]
    sq = (h - targets.to(h)) ** 2
    k = log_ratio.shape[0]
    if weights is None:
        iu = torch.triu_indices(k, k, offset=1)
        return sq[iu[0], iu[1]].mean()
    w = weights.to(h)
    return (w * sq).sum() / w.sum()


def offline_ipo_loss(log_ratio_chosen, log_ratio_rejected, tau: float):
    """IPO on fixed (chosen, rejected) pairs: (h - 1/(2 tau))^2, averaged."""
    return ((log_ratio_chosen - log_ratio_rejected - 0.5 / tau) ** 2).mean()


def response_mask(responses, eos_token_id: int):
    """1 for response tokens up to and including the first EOS, 0 after it
    (which also covers padding, whether or not pad == eos)."""
    is_eos = (responses == eos_token_id).long()
    after_eos = (is_eos.cumsum(dim=1) - is_eos) > 0
    return (~after_eos).long()


def sequence_logprobs(model, query, responses, mask):
    """Sum of log pi(response token | prefix) over unmasked response tokens.
    query: 1-D prompt ids shared by the batch; responses, mask: [B, R]."""
    import torch

    b, r = responses.shape
    p = query.shape[0]
    input_ids = torch.cat([query.unsqueeze(0).expand(b, p), responses], dim=1)
    attn = torch.cat([torch.ones(b, p, dtype=mask.dtype, device=mask.device), mask], dim=1)
    logits = model(input_ids=input_ids, attention_mask=attn).logits[:, p - 1:-1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok = logp.gather(-1, responses.unsqueeze(-1)).squeeze(-1)
    return (tok * mask).sum(dim=1)


def log_ratios(policy, query, responses, mask, *, with_grad: bool):
    """log pi(y) - log ref(y) per row; the reference is the adapter switched off."""
    import torch

    with torch.no_grad(), policy.disable_adapter():
        ref = sequence_logprobs(policy, query, responses, mask)
    with torch.set_grad_enabled(with_grad):
        pol = sequence_logprobs(policy, query, responses, mask)
    return pol - ref


def ipo_step(policy, optimizer, query, responses, mask, loss_fn, *,
             mini_batch_size: int, max_grad_norm: float) -> dict:
    """One optimiser step on a loss of the batch's log-ratios. The pairwise loss
    couples every row, so with several mini-batches it runs in two passes:
    log-ratios without grad give dL/dd, then each mini-batch backpropagates
    sum(dL/dd_i * d_i), which has the same gradient as the full-batch loss
    (up to LoRA dropout drawing different masks in the two passes)."""
    import torch

    policy.train()
    optimizer.zero_grad()
    n, mb = responses.shape[0], mini_batch_size
    chunks = [slice(s, s + mb) for s in range(0, n, mb)]

    def policy_logprobs(c, with_grad):
        with torch.set_grad_enabled(with_grad):
            return sequence_logprobs(policy, query, responses[c], mask[c])

    with torch.no_grad(), policy.disable_adapter():
        ref = torch.cat([sequence_logprobs(policy, query, responses[c], mask[c])
                         for c in chunks])
    if len(chunks) == 1:
        d = policy_logprobs(chunks[0], True) - ref
        loss = loss_fn(d)
        loss.backward()
        d_detached = d.detach()
    else:
        d_detached = torch.cat([policy_logprobs(c, False) for c in chunks]) - ref
        d_leaf = d_detached.clone().requires_grad_(True)
        loss = loss_fn(d_leaf)
        loss.backward()
        for c in chunks:
            (d_leaf.grad[c] * policy_logprobs(c, True)).sum().backward()
    params = [p for p in policy.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": float(grad_norm),
            "log_ratio_mean": float(d_detached.mean())}


def lr_multiplier(step: int, total_steps: int, decay_frac: float) -> float:
    """1 until the last `decay_frac` of training, then linear down to 0 at the
    final step. Damps the step-to-step jitter of the last iterate around the
    equilibrium, which sampled batches otherwise leave at full size."""
    if decay_frac <= 0:
        return 1.0
    start = total_steps * (1 - decay_frac)
    if step < start:
        return 1.0
    return max(0.0, (total_steps - step) / max(1.0, total_steps - start))


def build_policy(model_name: str, dtype: str, lora: dict, tokenizer, device: str):
    from peft import get_peft_model

    base = build_causal_lm(model_name, dtype, tokenizer)
    policy = get_peft_model(base, lora_config(lora, task_type="CAUSAL_LM"))
    policy.to(device)
    return policy


def generate(policy, query, n: int, max_new_tokens: int, pad_token_id: int,
             chunk: int = 64):
    """n sampled completions of the prompt, right-padded to max_new_tokens."""
    import torch

    policy.eval()
    out = []
    with torch.no_grad():
        for s in range(0, n, chunk):
            b = min(chunk, n - s)
            ids = query.unsqueeze(0).expand(b, -1)
            seq = policy.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  max_new_tokens=max_new_tokens, pad_token_id=pad_token_id,
                                  **tc.GEN_KWARGS)
            resp = seq[:, query.shape[0]:]
            pad = max_new_tokens - resp.shape[1]
            if pad:
                resp = torch.nn.functional.pad(resp, (0, pad), value=pad_token_id)
            out.append(resp)
    return torch.cat(out)


def generate_geometric_mixture(policy, query, n: int, max_new_tokens: int,
                               pad_token_id: int, eos_token_id: int, beta: float,
                               chunk: int = 64):
    """IPO-MD's sampler: n completions from pi^(1-beta) ref^beta, normalised per
    token, right-padded to max_new_tokens. Mixing towards the reference keeps
    every colour in the batch, so a policy that has collapsed onto one colour
    still sees the colour that beats it. No KV cache (the prefix is re-run each
    token): responses are a few tokens, and it avoids per-model cache types."""
    import torch

    policy.eval()
    out = []
    with torch.no_grad():
        for s in range(0, n, chunk):
            b = min(chunk, n - s)
            seq = query.unsqueeze(0).expand(b, -1).clone()
            done = torch.zeros(b, dtype=torch.bool, device=seq.device)
            for _ in range(max_new_tokens):
                pol = torch.log_softmax(policy(input_ids=seq).logits[:, -1].float(), -1)
                with policy.disable_adapter():
                    ref = torch.log_softmax(policy(input_ids=seq).logits[:, -1].float(), -1)
                probs = torch.softmax((1 - beta) * pol + beta * ref, -1)
                nxt = torch.multinomial(probs, 1).squeeze(-1)
                nxt = torch.where(done, torch.full_like(nxt, pad_token_id), nxt)
                done |= nxt == eos_token_id
                seq = torch.cat([seq, nxt.unsqueeze(-1)], dim=1)
            out.append(seq[:, query.shape[0]:])
    return torch.cat(out)


def encode_answers(tokenizer, colours: list[str], device: str):
    """`" <colour>."` as response ids, right-padded, with their mask: the
    canonical answer format the reward model is trained on too."""
    import torch

    ids = [tokenizer(" " + c + ".", add_special_tokens=False)["input_ids"] for c in colours]
    width = max(len(x) for x in ids)
    resp = torch.full((len(ids), width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros(len(ids), width, dtype=torch.long)
    for i, x in enumerate(ids):
        resp[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    return resp.to(device), mask.to(device)


def run(storage: Storage, profile: dict, config_name: str, method: str = "ipo",
        force: bool = False) -> dict:
    import torch

    online = method == "ipo"
    stage = f"train_{method}"
    tp = train_params(profile, method, config_name)
    seed = profile["data"]["seed"]
    params = {"config": config_name, "method": method, "model": profile["model"]["name"],
              "seed": seed, **tp}
    phash = manifest.params_hash(params)
    out = storage.stage_dir(stage, config_name)
    if not force and manifest.is_complete(out, phash):
        return {"skipped": True, "dir": str(out)}

    pop = CONFIGS[config_name]
    device = profile["model"]["device"]
    torch.manual_seed(seed)
    rng = random.Random(f"ipo-{seed}-{config_name}")
    rows = Storage.read_jsonl(storage.stage_dir("data_gen", config_name) / "triplets.jsonl")
    pref = empirical_preference(rows, pop.alternatives)
    tau, k = tp["tau"], tp["batch_size"]

    with stage_timer(out, stage, config=config_name, method=method,
                     gpu_type=profile["gpu"]["train"]):
        tokenizer = tc.build_tokenizer(profile["model"]["name"])
        policy = build_policy(profile["model"]["name"], profile["model"]["dtype"],
                              tp["lora"], tokenizer, device)
        # adam_beta1: momentum can make the last iterate orbit a preference
        # cycle instead of converging (tests/test_train_ipo.py)
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                                      lr=tp["learning_rate"], betas=(tp["adam_beta1"], 0.999))
        query = tokenizer(prompt_for(profile, pop.prompt_key),
                          return_tensors="pt")["input_ids"][0].to(device)
        max_new = profile["eval"]["max_new_tokens"]
        eos = tokenizer.eos_token_id

        steps_per_epoch = max(1, profile["data"]["n_datapoints"] // k)
        total_steps = tp["epochs"] * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: lr_multiplier(step, total_steps, tp["lr_decay_frac"]))
        loop = tc.TrainingLoop(out, policy, tokenizer, pop.alternatives,
                               tag=f"[{method}/{config_name}]", epochs=tp["epochs"],
                               steps_per_epoch=steps_per_epoch)
        mixture = tc.MixtureAccumulator(pop.alternatives)
        for epoch, it in loop.epochs_and_steps():
            if online and tp["mix_beta"] > 0:
                # IPO-MD: the batch (and so the metrics below) comes from the
                # geometric mixture with the reference, not from pi alone
                responses = generate_geometric_mixture(policy, query, k, max_new,
                                                       tokenizer.pad_token_id, eos,
                                                       tp["mix_beta"])
            else:
                responses = generate(policy, query, k, max_new, tokenizer.pad_token_id)
            texts = tc.decode_batch(tokenizer, responses)
            parsed = tc.parse_texts(texts, pop.alternatives)
            mixture.observe(parsed)
            if online:
                targets = pair_targets(parsed, pref, tau)
                stats = ipo_step(policy, optimizer, query, responses,
                                 response_mask(responses, eos),
                                 lambda d: online_ipo_loss(d, targets),
                                 mini_batch_size=tp["mini_batch_size"],
                                 max_grad_norm=tp["max_grad_norm"])
            else:
                batch = rng.sample(rows, min(k, len(rows)))
                resp, mask = encode_answers(
                    tokenizer, [r["chosen"] for r in batch] + [r["rejected"] for r in batch],
                    device)
                n = len(batch)
                stats = ipo_step(policy, optimizer, query, resp, mask,
                                 lambda d: offline_ipo_loss(d[:n], d[n:], tau),
                                 mini_batch_size=tp["mini_batch_size"],
                                 max_grad_norm=tp["max_grad_norm"])
            scheduler.step()
            # on the policy's own samples the mean log-ratio estimates
            # KL(pi || ref); on the offline pairs it is not a KL
            loop.after_step(epoch=epoch, it=it, parsed=parsed, texts=texts,
                            stats={"objective/kl": stats["log_ratio_mean"]} if online else {},
                            extra={"loss": round(stats["loss"], 4),
                                   "grad_norm": round(stats["grad_norm"], 4),
                                   "log_ratio_mean": round(stats["log_ratio_mean"], 4)})
        Storage.write_json(out / "mixture.json", mixture.result())
    manifest.write_manifest(out, phash,
                            ["checkpoint/", "metrics.jsonl", "samples.jsonl", "mixture.json"],
                            extra={"seed": seed, "base_parse_rate": loop.base_parse_rate})
    return {"skipped": False, "dir": str(out)}
