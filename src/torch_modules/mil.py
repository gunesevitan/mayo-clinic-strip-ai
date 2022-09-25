import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def init_weights(module,
                 linear_weight_init_type, linear_weight_init_args, linear_bias_init_type, linear_bias_init_args,
                 batch_normalization_weight_init_type=None, batch_normalization_weight_init_args=None,
                 batch_normalization_bias_init_type=None, batch_normalization_bias_init_args=None):

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


class ClassificationHead(nn.Module):

    def __init__(self,
                 input_features, intermediate_features, n_classes, pooling_type,
                 activation, activation_args, dropout_probability, initialization_args=None):

        super(ClassificationHead, self).__init__()

        self.pooling_type = pooling_type
        self.classifier = nn.Sequential(
            nn.Linear(input_features * 2 if pooling_type == 'concat' else input_features, intermediate_features, bias=True),
            getattr(nn, activation)(**activation_args),
            nn.Dropout(p=dropout_probability) if dropout_probability >= 0 else nn.Identity(),
            nn.Linear(intermediate_features, n_classes, bias=True),
            nn.Softmax(dim=-1) if n_classes > 1 else nn.Identity()
        )

        if initialization_args is not None:
            for layer in self.classifier:
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.BatchNorm1d):
                    init_weights(layer, **initialization_args)

    def forward(self, x):

        if self.pooling_type == 'avg':
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'max':
            x = F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'concat':
            x = torch.cat([
                F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1),
                F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
            ], dim=-1)

        output = self.classifier(x)

        return output


class MultiInstanceLearningModel(nn.Module):

    def __init__(self, n_instances, model_module, model_name, pretrained, head_args):

        super(MultiInstanceLearningModel, self).__init__()

        if model_module == 'timm':
            self.backbone = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                num_classes=head_args['n_classes']
            )
            n_classifier_features = self.backbone.get_classifier().in_features
        else:
            raise ValueError(f'Invalid model_module {model_module}')

        self.classification_head = ClassificationHead(input_features=n_classifier_features * n_instances, **head_args)

    def forward(self, x):

        # Stack instances on batch dimension before passing input to feature extractor
        input_batch_size, input_instance, input_channel, input_height, input_width = x.shape
        x = x.view(input_batch_size * input_instance, input_channel, input_height, input_width)
        x = self.backbone.forward_features(x)

        # Stack feature maps on channel dimension before passing feature maps to classification head
        feature_batch_size, feature_channel, feature_height, feature_width = x.shape
        x = x.contiguous().view(input_batch_size, feature_channel * input_instance, feature_height, feature_width)
        output = self.classification_head(x)

        return output
