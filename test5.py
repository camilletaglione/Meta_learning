import torch
import torch.nn.functional as F
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from collections import OrderedDict
from torchmeta.modules import MetaLinear


def update_parameters(model, loss, step_size=0.1):
    grads = torch.autograd.grad(loss, model.meta_parameters(), create_graph=True)

    params = OrderedDict()
    for (name, params), grad in zip(model.meta_named_parameters(), grads):
        params[name] = params - step_size * grad

    return params


dataset = omniglot("data", ways=5, shots=5, meta_train=True, download=True)
dataloader = BatchMetaDataLoader(dataset, batch_size=1, num_workers=4)

model = MetaLinear(28*28, 5)
optimizer = torch.optim.Adam(model.meta_parameters(), lr=1e-3)

for batch in dataloader:
    model.zero_grad()

    train_inputs, train_labels = batch['train']
    train_logits = model(train_inputs.squeeze().view((-1, 28 * 28)))
    inner_loss = F.cross_entropy(train_logits, train_labels.squeeze())

    params = update_parameters(model, inner_loss)
    test_inputs, test_label = batch['test']
    test_logits = model(test_inputs.squeezz().view((-1, 28 * 28)), params=params)
    outer_loss = F.cross_entropy(test_logits, test_label.squeeze())

    print('Loss: {0:.4f}'.format(outer_loss))
    outer_loss.backward()
    optimizer.step()

