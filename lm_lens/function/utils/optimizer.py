import torch
from torch.optim.optimizer import Optimizer


class Seme(Optimizer):
	r"""Implements Seme algorithm."""

	def __init__(self, params, lr=1e-2, weight_decay=0.0):
		"""Initialize the hyperparameters.

		Args:
			params (iterable): iterable of parameters to optimize or dicts defining
				parameter groups
			lr (float, optional): learning rate (default: 1e-4)
			weight_decay (float, optional): weight decay coefficient (default: 0)
		"""

		if not 0.0 <= lr:
			raise ValueError('Invalid learning rate: {}'.format(lr))
		defaults = dict(lr=lr, weight_decay=weight_decay)
		super().__init__(params, defaults)
		self.defaults['matrix_deltas'] = None

	# this is for the old implementation (3/4)
	def update_delta(self, matrix_deltas, trainable_layers=None):
		self.defaults['matrix_deltas'] = matrix_deltas
		self.defaults['trainable_layers'] = list(sorted(matrix_deltas.keys())) if trainable_layers is None else trainable_layers

	@torch.no_grad()
	def step(self, closure=None):
		"""Performs a single optimization step.

		Args:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.

		Returns:
			the loss.
		"""
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		trainable_layer_idx = 0
		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue

				grad = p.grad

				# # Perform stepweight decay
				# p.data.mul_(1 - group['lr'] * group['weight_decay'])

				assert self.defaults['matrix_deltas'] is not None
				matrix_deltas = self.defaults['matrix_deltas']
				trainable_layers = self.defaults['trainable_layers']
				layer_idx = trainable_layers[trainable_layer_idx]
				matrix_delta = matrix_deltas[layer_idx]  # .clone()  # .clone() may be important (...)
				trainable_layer_idx += 1

				# matrix_delta[matrix_delta.sign() != -grad.sign_()] = 0
				matrix_delta[matrix_delta.sign() != -grad.sign_()] *= -1
				p.add_(matrix_delta, alpha=group['lr'])

		return loss
