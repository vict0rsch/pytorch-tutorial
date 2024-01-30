# pytorch-tutorial
A self-contained python notebook, runnable on Google Colab, to discover Pytorch

<details>
  <summary>There are a few questions in the Notebook, click here to see the answers</summary>
  
  * `LinearModel.forward(...)`
    * `y = x @ self.weight.T + self.bias`
    * And variants (`torch.matmul(...)`, `torch.transpose(...)` etc.)
  * `custom_loss(...)`
    * `return torch.mean((predictions - ground_truth) ** 2)`
    * And variants (`(...).mean()`,`torch.pow(..., 2)` etc.)
  * `[:1]` vs `[0]`
    * The first one keeps a batch dimension, the second one removes the batch dimension. In general you want to keep it.
  * Example `MLP(nn.Module)` class

    ```python
    class MLP(nn.Module):
        def __init__(self, layer_sizes, activation=nn.LeakyReLU(), dropout=0.5):
            super().__init__()
            
            self.layers = []
            for k, ls in enumerate(layer_sizes[:-2]):
                self.layers.append(nn.Linear(ls, layer_sizes[k+1]))
                self.layers.append(activation)
                self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
            self.layers = nn.Sequential(*self.layers)
    
        def forward(self, x):
            return self.layers(x)
    
    def make_mlp(layer_sizes, activation=nn.LeakyReLU(), dropout=0.5):
        layers = []
        for k, ls in enumerate(layer_sizes[:-2]):
            layers.append(nn.Linear(ls, layer_sizes[k+1]))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        return nn.Sequential(*layers)

    ```
  * Section 3: fatal training issues:
    * Notice the validation accuracy is constant, your model is not learning.
    * The training metrics oscillate because of randomness (data loading, dropout etc.), which disappears at validation time.
    * Notice the missing `error.backward()`: no gradients are computed, so no update to the model weights.
    * `torch.nn.CrossEntropyLoss()` takes **unormalized** probabilities, *i.e.* there should not be anything after the last `Linear` layer (same for regression).
      * in the `CNN()` `mlp` `for` loop, indent the `activation` under `if i != (len(mlp_sizes) - 2):` because you don't want an activation for the last MLP layer.
    * Not _fatal_ but still an issue: the learning rate is too large. Try `lr = 0.001` or `lr = 0.0005`.
</details>
