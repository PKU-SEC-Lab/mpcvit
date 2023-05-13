from torch.hub import load_state_dict_from_url
import torch.nn as nn
from .utils.transformers import TransformerClassifier
from .utils.tokenizer import Tokenizer
from .utils.helpers import pe_check
from .utils.transformers_teacher import TransformerClassifier_teacher

try:
    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model

model_urls = {
}


class ViTLite(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 kernel_size=16,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 *args, **kwargs):
        super(ViTLite, self).__init__()
        assert img_size % kernel_size == 0, f"Image size ({img_size}) has to be" \
                                            f"divisible by patch size ({kernel_size})"
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=kernel_size,
                                   padding=0,
                                   max_pool=False,
                                   activation=None,
                                   n_conv_layers=1,
                                   conv_bias=True)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=False,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding
        )

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


class ViTLite_teacher(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 kernel_size=16,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 *args, **kwargs):
        super(ViTLite_teacher, self).__init__()
        assert img_size % kernel_size == 0, f"Image size ({img_size}) has to be" \
                                            f"divisible by patch size ({kernel_size})"
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=kernel_size,
                                   padding=0,
                                   max_pool=False,
                                   activation=None,
                                   n_conv_layers=1,
                                   conv_bias=True)

        self.classifier = TransformerClassifier_teacher(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=False,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding
        )

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


def _vit_lite(arch, pretrained, progress,
              num_layers, num_heads, mlp_ratio, embedding_dim,
              kernel_size=4, positional_embedding='learnable',
              *args, **kwargs):
    ## note: the arg 'positional_embedding' should be above
    model = ViTLite(num_layers=num_layers,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    embedding_dim=embedding_dim,
                    kernel_size=kernel_size,
                    positional_embedding='learnable',
                    *args, **kwargs)

    if pretrained and arch in model_urls:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        if positional_embedding == 'learnable':
            state_dict = pe_check(model, state_dict)
        elif positional_embedding == 'sine':
            state_dict['classifier.positional_emb'] = model.state_dict()['classifier.positional_emb']
        state_dict = pe_check(model, state_dict)
        model.load_state_dict(state_dict)
    return model


def _vit_lite_teacher(arch, pretrained, progress,
              num_layers, num_heads, mlp_ratio, embedding_dim,
              kernel_size=4, positional_embedding='learnable',
              *args, **kwargs):
    ## note: the arg 'positional_embedding' should be above
    model = ViTLite_teacher(num_layers=num_layers,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    embedding_dim=embedding_dim,
                    kernel_size=kernel_size,
                    positional_embedding='learnable',
                    *args, **kwargs)

    if pretrained and arch in model_urls:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        if positional_embedding == 'learnable':
            state_dict = pe_check(model, state_dict)
        elif positional_embedding == 'sine':
            state_dict['classifier.positional_emb'] = model.state_dict()['classifier.positional_emb']
        state_dict = pe_check(model, state_dict)
        model.load_state_dict(state_dict)
    return model


def vit_2(*args, **kwargs):
    return _vit_lite(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                     *args, **kwargs)


def vit_4(*args, **kwargs):
    return _vit_lite(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                     *args, **kwargs)


def vit_6(*args, **kwargs):
    return _vit_lite(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                     *args, **kwargs)


# ViT arch for cifar-10/100
def vit_7(*args, **kwargs):
    return _vit_lite(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                     *args, **kwargs)


def vit_7_wider(*args, **kwargs):
    return _vit_lite(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=384,
                     *args, **kwargs)


def vit_7_teacher(*args, **kwargs):
    return _vit_lite_teacher(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                     *args, **kwargs)


## ViT arch for tiny-imagenet
def vit_9(*args, **kwargs):
    return _vit_lite(num_layers=9, num_heads=12, mlp_ratio=2, embedding_dim=192,
                     *args, **kwargs)


def vit_9_teacher(*args, **kwargs):
    return _vit_lite_teacher(num_layers=9, num_heads=12, mlp_ratio=2, embedding_dim=192,
                     *args, **kwargs)


## ViT arch for imagenet
def vit_12(*args, **kwargs):
    return _vit_lite(num_layers=12, num_heads=12, mlp_ratio=4, embedding_dim=192,
                     *args, **kwargs)


def vit_12_teacher(*args, **kwargs):
    return _vit_lite_teacher(num_layers=12, num_heads=12, mlp_ratio=4, embedding_dim=192,
                     *args, **kwargs)


@register_model
def vit_2_4_32(pretrained=False, progress=False,
               img_size=32, positional_embedding='learnable', num_classes=10,
               *args, **kwargs):
    return vit_2('vit_2_4_32', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def vit_2_4_32_sine(pretrained=False, progress=False,
                    img_size=32, positional_embedding='sine', num_classes=10,
                    *args, **kwargs):
    return vit_2('vit_2_4_32_sine', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def vit_4_4_32(pretrained=False, progress=False,
               img_size=32, positional_embedding='learnable', num_classes=10,
               *args, **kwargs):
    return vit_4('vit_4_4_32', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def vit_4_4_32_sine(pretrained=False, progress=False,
                    img_size=32, positional_embedding='sine', num_classes=10,
                    *args, **kwargs):
    return vit_4('vit_4_4_32_sine', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def vit_6_4_32(pretrained=False, progress=False,
               img_size=32, positional_embedding='learnable', num_classes=10,
               *args, **kwargs):
    return vit_6('vit_6_4_32', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def vit_6_4_32_sine(pretrained=False, progress=False,
                    img_size=32, positional_embedding='sine', num_classes=10,
                    *args, **kwargs):
    return vit_6('vit_6_4_32_sine', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def vit_7_4_32(pretrained=False, progress=False,
               img_size=32, positional_embedding='learnable', num_classes=10,
               *args, **kwargs):
    return vit_7('vit_7_4_32', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)  # kernel_size is changed by patch_size


@register_model
def vit_7_4_32_wider(pretrained=False, progress=False,
               img_size=32, positional_embedding='learnable', num_classes=10,
               *args, **kwargs):
    return vit_7_wider('vit_7_4_320', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)  # kernel_size is changed by patch_size


@register_model
def vit_7_4_32_sine(pretrained=False, progress=False,
                    img_size=32, positional_embedding='sine', num_classes=10,
                    *args, **kwargs):
    return vit_7('vit_7_4_32_sine', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def vit_7_4_32_teacher(pretrained=False, progress=False,
               img_size=32, positional_embedding='learnable', num_classes=10,
               *args, **kwargs):
    return vit_7_teacher('vit_7_4_32_teacher', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)  # kernel_size is changed by patch_size
    

@register_model
def vit_9_12_64(pretrained=False, progress=False,
               img_size=64, positional_embedding='learnable', num_classes=200,
               *args, **kwargs):
    return vit_9('vit_9_12_64', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)  # kernel_size is changed by patch_size


@register_model
def vit_9_12_64_teacher(pretrained=False, progress=False,
               img_size=64, positional_embedding='learnable', num_classes=200,
               *args, **kwargs):
    return vit_9_teacher('vit_9_12_64_teacher', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)  # kernel_size is changed by patch_size


@register_model
def vit_12_12_224(pretrained=False, progress=False,
               img_size=224, positional_embedding='learnable', num_classes=1000,
               *args, **kwargs):
    return vit_12('vit_12_12_224', pretrained, progress,
                 kernel_size=16,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)  # kernel_size is changed by patch_size


@register_model
def vit_12_12_224_teacher(pretrained=False, progress=False,
               img_size=224, positional_embedding='learnable', num_classes=1000,
               *args, **kwargs):
    return vit_12_teacher('vit_12_12_224_teacher', pretrained, progress,
                 kernel_size=16,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)  # kernel_size is changed by patch_size