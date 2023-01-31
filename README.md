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
  * Section 3: fatal training issues
    * Notice the validation accuracy is constant, your model is not learning
    * The training metrics oscillate because of randomness (data loading, dropout etc.), which disappears at validation time
    * Notice the missing `error.backward()`: no gradients are computed, so no update to the model weights
    * `torch.nn.CrossEntropyLoss()` takes **unormalized** probabilities, *i.e.* there should not be anything after the last `Linear` layer (same for regression)
</details>
