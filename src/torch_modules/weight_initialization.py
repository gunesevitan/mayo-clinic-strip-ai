import torch.nn as nn


def initialize_weights(
        module,
        linear_weight_init_type, linear_weight_init_args, linear_bias_init_type, linear_bias_init_args,
        batch_normalization_weight_init_type=None, batch_normalization_weight_init_args=None,
        batch_normalization_bias_init_type=None, batch_normalization_bias_init_args=None
):

    """
    Initialize weights and biases of given layers with specified configurations

    Parameters
    ----------
    module (torch.nn.Module): Layer
    linear_weight_init_type (str): Weight initialization method of the linear layer
    linear_weight_init_args (dict): Weight initialization arguments of the linear layer
    linear_bias_init_type (str): Bias initialization method of the linear layer
    linear_bias_init_args (dict): Bias initialization arguments of the linear layer
    batch_normalization_weight_init_type (str): Weight initialization method of the batch normalization layer
    batch_normalization_weight_init_args (dict): Weight initialization arguments of the batch normalization layer
    batch_normalization_bias_init_type (str): Bias initialization method of the batch normalization layer
    batch_normalization_bias_init_args (dict): Bias initialization arguments of the batch normalization layer
    """

    if isinstance(module, nn.Linear):
        # Initialize weights of linear layer
        if linear_weight_init_type == 'uniform':
            nn.init.uniform_(
                module.weight,
                a=linear_weight_init_args['a'],
                b=linear_weight_init_args['b']
            )
        elif linear_weight_init_type == 'normal':
            nn.init.normal_(
                module.weight,
                mean=linear_weight_init_args['mean'],
                std=linear_weight_init_args['std']
            )
        elif linear_weight_init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(
                module.weight,
                gain=nn.init.calculate_gain(
                    nonlinearity=linear_weight_init_args['nonlinearity'],
                    param=linear_weight_init_args['nonlinearity_param']
                )
            )
        elif linear_weight_init_type == 'xavier_normal':
            nn.init.xavier_normal_(
                module.weight,
                gain=nn.init.calculate_gain(
                    nonlinearity=linear_weight_init_args['nonlinearity'],
                    param=linear_weight_init_args['nonlinearity_param']
                )
            )
        elif linear_weight_init_type == 'kaiming_uniform':
            nn.init.kaiming_uniform_(
                module.weight,
                a=linear_weight_init_args['nonlinearity_param'],
                mode=linear_weight_init_args['mode'],
                nonlinearity=linear_weight_init_args['nonlinearity']
            )
        elif linear_weight_init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(
                module.weight,
                a=linear_weight_init_args['nonlinearity_param'],
                mode=linear_weight_init_args['mode'],
                nonlinearity=linear_weight_init_args['nonlinearity']
            )
        elif linear_weight_init_type == 'orthogonal':
            nn.init.orthogonal_(
                module.weight,
                gain=nn.init.calculate_gain(
                    nonlinearity=linear_weight_init_args['nonlinearity'],
                    param=linear_weight_init_args['nonlinearity_param']
                )
            )
        elif linear_weight_init_type == 'sparse':
            nn.init.sparse_(
                module.weight,
                sparsity=linear_weight_init_args['sparsity'],
                std=linear_weight_init_args['std']
            )
        # Initialize biases of Linear layer
        if module.bias is not None:
            if linear_bias_init_type == 'uniform':
                nn.init.uniform_(
                    module.bias,
                    a=linear_bias_init_args['a'],
                    b=linear_bias_init_args['b']
                )
            elif linear_bias_init_type == 'normal':
                nn.init.normal_(
                    module.bias,
                    mean=linear_bias_init_args['mean'],
                    std=linear_bias_init_args['std']
                )

    elif isinstance(module, nn.BatchNorm1d):
        # Initialize weights of batch normalization layer
        if batch_normalization_weight_init_type is not None:
            if batch_normalization_weight_init_type == 'uniform':
                nn.init.uniform_(
                    module.weight,
                    a=batch_normalization_weight_init_args['a'],
                    b=batch_normalization_weight_init_args['b']
                )
            elif batch_normalization_weight_init_type == 'normal':
                nn.init.normal_(
                    module.weight,
                    mean=batch_normalization_weight_init_args['mean'],
                    std=batch_normalization_weight_init_args['std']
                )
            elif batch_normalization_weight_init_type == 'constant':
                nn.init.constant_(
                    module.weight,
                    val=batch_normalization_weight_init_args['val'],
                )
        # Initialize biases of batch normalization layer
        if batch_normalization_bias_init_type is not None:
            if batch_normalization_bias_init_type == 'uniform':
                nn.init.uniform_(
                    module.bias,
                    a=batch_normalization_bias_init_args['a'],
                    b=batch_normalization_bias_init_args['b']
                )
            elif batch_normalization_bias_init_type == 'normal':
                nn.init.normal_(
                    module.bias,
                    mean=batch_normalization_bias_init_args['mean'],
                    std=batch_normalization_bias_init_args['std']
                )
            elif batch_normalization_bias_init_type == 'constant':
                nn.init.constant_(
                    module.bias,
                    val=batch_normalization_bias_init_args['val'],
                )
