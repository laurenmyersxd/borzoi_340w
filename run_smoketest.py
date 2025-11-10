import os, glob, numpy as np
import tensorflow as tf
import borzoi            # ensure package is loaded
import borzoi.layers     # <-- important: registers custom layers with Keras


# ------------- config (auto-finds the 4 folds you just downloaded) -------------
BORZOI_DIR = os.environ.get("BORZOI_DIR", os.path.abspath("."))
model_glob = os.path.join(BORZOI_DIR, "examples", "saved_models", "f3c*", "train", "model0_best.h5")
model_paths = sorted(glob.glob(model_glob))
assert model_paths, f"No models found with glob: {model_glob}"

print(f"[info] found {len(model_paths)} models:")
for p in model_paths:
    print("   -", p)

# ------------- helper: make a valid one-hot DNA tensor for the model -------------
def make_onehot_like(shape):
    """
    Make a 1-hot nucleotide tensor matching the model's input shape.
    We assume input ends with 4 channels (A,C,G,T).
    """
    x = np.zeros(shape, dtype=np.float32)
    # place a simple repeating pattern over the sequence length
    # shape is typically (batch, seq_len, 4); if not, we adapt below.
    if len(shape) == 3 and shape[-1] == 4:
        _, L, C = shape
        idx = np.arange(L) % C
        x[np.arange(1), np.arange(L), idx] = 1.0  # batch=1
    elif len(shape) == 4 and shape[-1] == 4:
        # sometimes (batch, H, W, C); treat H as sequence length
        _, H, W, C = shape
        # collapse W if it’s 1; otherwise tile a simple pattern
        if W == 1:
            idx = np.arange(H) % C
            x[0, np.arange(H), 0, idx] = 1.0
        else:
            # fill a diagonal-ish stripe
            for i in range(H):
                x[0, i, i % W, (i % C)] = 1.0
    else:
        raise ValueError(f"Unexpected input shape {shape}; last dim must be 4 for A/C/G/T.")
    return x

# ------------- load each model, infer input shape, run predict -------------------
ok = True
for mpath in model_paths:
    print(f"\n[load] {mpath}")
    model = tf.keras.models.load_model(mpath, compile=False)
    ishape = model.input_shape
    if isinstance(ishape, (list, tuple)) and isinstance(ishape[0], (list, tuple)):
        # handle multi-input models by taking the first tensor input
        in_shape = tuple(int(d) if d is not None else 1 for d in ishape[0])
    else:
        in_shape = tuple(int(d) if d is not None else 1 for d in ishape)

    # ensure batch dimension = 1
    if in_shape[0] is None:
        in_shape = (1,) + in_shape[1:]
    elif in_shape[0] != 1:
        in_shape = (1,) + in_shape[1:]

    print("[shape] model.input_shape:", model.input_shape, "=> using", in_shape)

    # dummy one-hot forward (smoke)
    x = make_onehot_like(in_shape)
    y = model.predict(x, verbose=0)
    # print concise info
    if isinstance(y, (list, tuple)):
        outs = [np.array(o).shape for o in y]
    else:
        outs = [np.array(y).shape]
    print("[ok] forward pass complete. output shapes:", outs)

print("\n✅ smoketest finished without errors.")
