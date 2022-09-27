import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .weight_initialization import initialize_weights


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
            for module in self.classifier:
                if isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm1d):
                    initialize_weights(module=module, **initialization_args)

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
