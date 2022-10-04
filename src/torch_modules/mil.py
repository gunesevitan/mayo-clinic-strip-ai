import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .weight_initialization import initialize_weights


class ConvolutionalClassificationHead(nn.Module):

    def __init__(self,
                 input_features, intermediate_features, n_classes, pooling_type,
                 activation, activation_args, dropout_probability=0., batch_normalization=False,
                 initialization_args=None):

        super(ConvolutionalClassificationHead, self).__init__()

        self.pooling_type = pooling_type
        self.classifier = nn.Sequential(
            nn.Linear(input_features * 2 if pooling_type == 'concat' else input_features, intermediate_features, bias=True),
            getattr(nn, activation)(**activation_args),
            nn.BatchNorm1d(num_features=intermediate_features) if batch_normalization else nn.Identity(),
            nn.Dropout(p=dropout_probability) if dropout_probability >= 0. else nn.Identity(),
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


class TransformerClassificationHead(nn.Module):

    def __init__(self,
                 input_features, intermediate_features, n_classes,
                 activation, activation_args, dropout_probability=0., batch_normalization=False,
                 initialization_args=None):

        super(TransformerClassificationHead, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_features, intermediate_features, bias=True),
            getattr(nn, activation)(**activation_args),
            nn.BatchNorm1d(num_features=intermediate_features) if batch_normalization else nn.Identity(),
            nn.Dropout(p=dropout_probability) if dropout_probability >= 0. else nn.Identity(),
            nn.Linear(intermediate_features, n_classes, bias=True),
            nn.Softmax(dim=-1) if n_classes > 1 else nn.Identity()
        )

        if initialization_args is not None:
            for module in self.classifier:
                if isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm1d):
                    initialize_weights(module=module, **initialization_args)

    def forward(self, x):

        output = self.classifier(x)

        return output


class ConvolutionalMultiInstanceLearningModel(nn.Module):

    def __init__(self, n_instances, model_name, pretrained, freeze_parameters, aggregation, head_class, head_args):

        super(ConvolutionalMultiInstanceLearningModel, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=head_args['n_classes']
        )

        if freeze_parameters is not None:
            # Freeze all parameters in backbone
            if freeze_parameters == 'all':
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False
            else:
                # Freeze specified parameters in backbone
                for group in freeze_parameters:
                    if isinstance(self.backbone, timm.models.DenseNet):
                        for parameter in self.backbone.features[group].parameters():
                            parameter.requires_grad = False
                    elif isinstance(self.backbone, timm.models.EfficientNet):
                        for parameter in self.backbone.blocks[group].parameters():
                            parameter.requires_grad = False

        self.aggregation = aggregation
        n_classifier_features = self.backbone.get_classifier().in_features
        input_features = (n_classifier_features * n_instances) if self.aggregation == 'concat' else n_classifier_features
        self.classification_head = eval(head_class)(input_features=input_features, **head_args)

    def forward(self, x):

        # Stack instances on batch dimension before passing input to feature extractor
        input_batch_size, input_instance, input_channel, input_height, input_width = x.shape
        x = x.view(input_batch_size * input_instance, input_channel, input_height, input_width)
        x = self.backbone.forward_features(x)
        feature_batch_size, feature_channel, feature_height, feature_width = x.shape

        if self.aggregation == 'avg':
            # Average feature maps of multiple instances
            x = x.contiguous().view(input_batch_size, input_instance, feature_channel, feature_height, feature_width)
            x = torch.mean(x, dim=1)
        elif self.aggregation == 'max':
            # Max feature maps of multiple instances
            x = x.contiguous().view(input_batch_size, input_instance, feature_channel, feature_height, feature_width)
            x = torch.max(x, dim=1)
        elif self.aggregation == 'concat':
            # Stack feature maps on channel dimension before passing feature maps to classification head
            x = x.contiguous().view(input_batch_size, input_instance * feature_channel, feature_height, feature_width)

        output = self.classification_head(x)
        return output


class TransformerMultiInstanceLearningModel(nn.Module):

    def __init__(self, n_instances, model_name, pretrained, freeze_parameters, head_class, head_args):

        super(TransformerMultiInstanceLearningModel, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=head_args['n_classes']
        )

        if freeze_parameters is not None:
            # Freeze all parameters in backbone
            if freeze_parameters == 'all':
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False

        n_classifier_features = self.backbone.get_classifier().in_features
        self.backbone.head = nn.Identity()
        self.classification_head = eval(head_class)(input_features=n_classifier_features * n_instances, **head_args)

    def forward(self, x):

        # Stack instances on batch dimension before passing input to feature extractor
        input_batch_size, input_instance, input_channel, input_height, input_width = x.shape
        x = x.view(input_batch_size * input_instance, input_channel, input_height, input_width)
        x = self.backbone(x)

        # Stack feature maps on channel dimension before passing feature maps to classification head
        feature_batch_size, feature_count = x.shape
        x = x.contiguous().view(input_batch_size, feature_count * input_instance)
        output = self.classification_head(x)

        return output
