import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import numpy as np
from typing import Tuple
from orbax.checkpoint import CheckpointManager, Checkpointer, PyTreeCheckpointHandler, CheckpointManagerOptions
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import datetime
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score 

def _loss_fn(params, apply_fn, inputs, y, rng, train:bool):
    logits = apply_fn(
        params, 
        *inputs, 
        train=train, 
        rngs={"dropout": rng} if train else None
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss, logits

@jax.jit
def train_step(state: train_state.TrainState, inputs: Tuple[jnp.ndarray, jnp.ndarray], y: jnp.ndarray, rng):
    def loss_fn(p):
        loss, _ =  _loss_fn(p, state.apply_fn, inputs, y, rng, train=True)
        return loss
        
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state: train_state.TrainState, inputs: Tuple[jnp.ndarray, jnp.ndarray], y: jnp.ndarray):
    loss, logits = _loss_fn(
        state.params,
        state.apply_fn,
        inputs,
        y,
        rng=None,
        train=False
    )
    y_preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(y_preds == y)
    return loss, acc, y_preds
    
class Trainer:
    def __init__(self, model, rng=jax.random.PRNGKey(0), learning_rate=1e-4, log_dir="./deep_learning/logs", ckpt_dir='./deep_learning/checkpoints'):
        self.model = model
        self.learning_rate = learning_rate
        self.state = None
        self.rng = rng
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        log_dir = Path(log_dir).resolve() / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
        
        ckpt_path = Path(ckpt_dir).resolve() / timestamp
        ckpt_path.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        
        self.checkpointer = Checkpointer(PyTreeCheckpointHandler())
        manager_options = CheckpointManagerOptions(max_to_keep=10)
        self.ckpt_manager = CheckpointManager(ckpt_path, self.checkpointer, options=manager_options)
        
    def compile(self, rng, input_shape):
        print("Compiling...")
        dummy_inputs = [jnp.ones((1, 10, shape[2])) for shape in input_shape]
        params = self.model.init(rng, *dummy_inputs, train=False)
        
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(self.learning_rate, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01)
        )
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, 
            params=params, 
            tx=tx
        )
    
    def fit(self, inputs, y, epochs=10, batch_size=8, validation_data=None):
        assert self.state is not None, "Call compile() first"
        
        n_samples = y.shape[0]
        for epoch in range(epochs):
            
            perm = np.random.permutation(n_samples)
            inputs_shuffled = [x[perm] for x in inputs]
            y_shuffled = y[perm]
            
            # Mini batch training - full batch causes mem issues
            epoch_loss = 0.0
            epoch_acc = 0.0
            pbar = tqdm(range(0, n_samples, batch_size), desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            for i in pbar:
                self.rng, step_rng = jax.random.split(self.rng)
                batch_inputs = [x[i:i+batch_size] for x in inputs_shuffled]
                batch_y = y_shuffled[i:i+batch_size]
                self.state, loss = train_step(self.state, batch_inputs, batch_y, step_rng)
                
                # Accuracy per batch
                logits = self.state.apply_fn(
                    self.state.params,
                    *batch_inputs,
                    train=False
                )
                batch_preds = jnp.argmax(logits, axis=1)
                batch_acc = jnp.mean(batch_preds == batch_y)
                
                epoch_loss += float(loss) * batch_y.shape[0]
                epoch_acc += float(batch_acc) * batch_y.shape[0]
                pbar.set_postfix({'Loss': epoch_loss / (i + batch_size), 'Acc': epoch_acc / (i + batch_size)})
               
            epoch_loss /= n_samples
            epoch_acc /= n_samples

            # Validation
            if validation_data is not None:
                (x_bp_val, x_ecg_val), y_val = validation_data
                x_bp_val = jnp.asarray(x_bp_val)
                x_ecg_val = jnp.asarray(x_ecg_val)
                
                val_loss = 0.0
                val_acc = 0.0
                n_val = y_val.shape[0]
                preds = jnp.array([], dtype=jnp.int32)
                
                for j in range(0, n_val, batch_size):
                    xb = x_bp_val[j:j+batch_size]
                    xe = x_ecg_val[j:j+batch_size]
                    yb = y_val[j:j+batch_size]
                    yb = yb.reshape(-1)
                    # Loss acc
                    loss, acc, y_pred = eval_step(self.state, (xb, xe), yb)
                    val_loss += float(loss) * yb.shape[0]
                    val_acc += float(acc) * yb.shape[0]
                    
                    preds = jnp.concatenate([preds, y_pred])
                    
                val_loss /= n_val
                val_acc /= n_val   
                
                # F1 - Recall - Precision
                precision = precision_score(y_true=y_val, y_pred=preds)
                recall = recall_score(y_true=y_val, y_pred=preds)
                f1 = f1_score(y_true=y_val, y_pred=preds)
                
                # Per class
                f1_per_class = f1_score(y_true=y_val, y_pred=preds, average=None)
                  
            # Logging
            self.writer.add_scalar("train/loss", epoch_loss, epoch)
            self.writer.add_scalar("train/accuracy", epoch_acc, epoch)
            if val_loss is not None:
                self.writer.add_scalar("val/loss", val_loss, epoch)
                self.writer.add_scalar("val/accuracy", val_acc, epoch)
                
            if f1 is not None:
                self.writer.add_scalar("val/precision", precision, epoch)
                self.writer.add_scalar("val/recall", recall, epoch)
                self.writer.add_scalar("val/f1", f1, epoch)   
                self.writer.add_scalar(f"val/f1 for class 0 ", f1_per_class[0], epoch) 
                self.writer.add_scalar(f"val/f1 for class 1 ", f1_per_class[1], epoch) 

            
            print(f"Epoch {epoch +1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}", end="")
            if val_loss is not None:
                print(f", val_loss: {val_loss:.4}, val acc: {val_acc:.4f}")
            else:
                print()
                
            self.ckpt_manager.save(epoch, (self.state, {"train_loss": epoch_loss, "val_loss": val_loss, "val_acc": val_acc}))

    def save(self, path):
        checkpoints.save_checkpoint(ckpt_dir=path, target=self.state.params, step=0, overwrite=True)