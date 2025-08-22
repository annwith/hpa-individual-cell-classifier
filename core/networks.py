# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter

from tools.ai.torch_utils import (gap2d, set_trainable_layers)


class FixedBatchNorm(nn.BatchNorm2d):

  def forward(self, x):
    return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)


def patch_conv_in_channels(model, layer_name, new_in_channels, copying_channel=0):
  layer = getattr(model, layer_name)  # layer = model.conv1
  # new_layer = layer.clone().detach()

  if isinstance(layer, nn.Sequential):
    cv_layers = list(layer.children())
    cv = cv_layers[0]
  else:
    cv = layer

  if not isinstance(cv, nn.Conv2d):
    raise ValueError(f"Cannot extract Conv2d from {cv}.")

  new_cv = nn.Conv2d(
    in_channels=new_in_channels,
    out_channels=cv.out_channels,
    kernel_size=cv.kernel_size,
    stride=cv.stride,
    padding=cv.padding,
    bias=cv.bias).requires_grad_()

  with torch.no_grad():
    new_cv.weight[:, :cv.in_channels, :, :] = cv.weight.data

    for i in range(new_in_channels - cv.in_channels):
        channel = cv.in_channels + i
        new_cv.weight[:, channel:channel+1, :, :] = cv.weight[:, copying_channel:copying_channel+1, : :].data
  new_cv.weight = nn.Parameter(new_cv.weight)

  if isinstance(layer, nn.Sequential):
    new_cv = nn.Sequential(new_cv, *cv_layers[1:])

  setattr(model, layer_name, new_cv)  # model.conv1 = new_layer


def build_backbone(name, dilated, strides, norm_fn, weights='imagenet', channels=3, **kwargs):
  if 'resnet38d' == name:
    from .backbones.arch_resnet import resnet38d

    model = resnet38d.ResNet38d()
    state_dict = resnet38d.convert_mxnet_to_torch('./experiments/models/resnet_38d.params')
    model.load_state_dict(state_dict, strict=True)

    if channels != 3:
      patch_conv_in_channels(model, "conv1a", channels)

    stages = (
      nn.Sequential(model.conv1a, model.b2, model.b2_1, model.b2_2),
      nn.Sequential(model.b3, model.b3_1, model.b3_2),
      nn.Sequential(model.b4, model.b4_1, model.b4_2, model.b4_3, model.b4_4, model.b4_5),
      nn.Sequential(model.b5, model.b5_1, model.b5_2),
      nn.Sequential(model.b6, model.b7, model.bn7, nn.ReLU()),
    )

  elif "swin" in name:
    if "swinv2" in name:
      from .backbones import swin_transformer_v2 as swin_mod

      model_fn = getattr(swin_mod, name)
      model = model_fn(**kwargs)

      if channels != 3:
        model.patch_embed = channels
        patch_conv_in_channels(model.patch_embed, "proj", channels)

      stages = (
        nn.Sequential(model.patch_embed, model.pos_drop),
        *model.layers[:3],
        nn.Sequential(model.layers[3], model.norm, swin_mod.TransposeLayer((1, 2))),
      )
    else:
      from .backbones import swin_transformer as swin_mod

      model_fn = getattr(swin_mod, name)
      model = model_fn(in_chans=in_channels, out_indices=(3,), **kwargs)

      stages = (nn.Sequential(model.patch_embed, model.pos_drop), *model.layers)

    if weights and weights != 'imagenet':
      print(f'loading weights from {weights}')
      checkpoint = torch.load(weights, map_location="cpu")
      checkpoint = checkpoint["model"]
      del checkpoint["head.weight"]
      del checkpoint["head.bias"]

      model.load_state_dict(checkpoint, strict=False)

  elif "mit" in name:
    from .backbones import mix_transformer

    model_fn = getattr(mix_transformer, name)
    model = model_fn(**kwargs)

    if weights and weights != "imagenet":
      print(f'loading weights from {weights}')
      checkpoint = torch.load(weights, map_location="cpu")
      del checkpoint["head.weight"]
      del checkpoint["head.bias"]

      model.load_state_dict(checkpoint)

    if channels != 3:
      patch_conv_in_channels(model.patch_embed1, "proj", channels)

    stages = (
      nn.Sequential(model.patch_embed1, model.block1, model.norm1),
      nn.Sequential(model.patch_embed2, model.block2, model.norm2),
      nn.Sequential(model.patch_embed3, model.block3, model.norm3),
      nn.Sequential(model.patch_embed4, model.block4, model.norm4),
    )

  else:
    if 'resnet' in name:
      from .backbones.arch_resnet import resnet
      if dilated:
        strides = strides or (1, 2, 1, 1)
        dilations = (1, 1, 2, 4)
      else:
        strides = strides or (1, 2, 2, 1)
        dilations = (1, 1, 1, 2)
      model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic[name], strides=strides, dilations=dilations, batch_norm_fn=norm_fn)

      if weights == 'imagenet':
        print(f'loading weights from {resnet.urls_dic[name]}')
        state_dict = model_zoo.load_url(resnet.urls_dic[name])
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')

        model.load_state_dict(state_dict)

      if channels != 3:
        patch_conv_in_channels(model, "conv1", channels)

    elif 'resnest' in name:
      from .backbones.arch_resnest import resnest
      dilation = 4 if dilated else 2

      pretrained = weights == "imagenet"
      model_fn = getattr(resnest, name)
      model = model_fn(pretrained=pretrained, dilated=dilated, dilation=dilation, norm_layer=norm_fn)

      if pretrained:
        print(f'loading weights from {resnest.resnest_model_urls[name]}')

      if weights and weights != 'imagenet':
        print(f'loading weights from {weights}')
        checkpoint = torch.load(weights, map_location="cpu")
        # model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.load_state_dict(checkpoint, strict=False)

      if channels != 3:
        print("Modify input layer to receive 4 channels images.")
        patch_conv_in_channels(model, "conv1", channels)

      del model.avgpool
      del model.fc
    elif 'res2net' in name:
      from .backbones.res2net import res2net_v1b

      pretrained = weights == "imagenet"
      model_fn = getattr(res2net_v1b, name)
      model = model_fn(pretrained=pretrained, strides=strides or (1, 2, 2, 2), norm_layer=norm_fn)
      if channels != 3:
        patch_conv_in_channels(model, "conv1", channels)  # FIX: conv1 is Sequential
      if pretrained:
        print(f'loading {weights} pretrained weights')

      del model.avgpool
      del model.fc

    # if weights and weights != 'imagenet':
    #   print(f'loading weights from {weights}')
    #   checkpoint = torch.load(weights, map_location="cpu")
    #   model.load_state_dict(checkpoint['state_dict'], strict=False)
    #   model.load_state_dict(checkpoint, strict=False)

    stages = (
      nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),
      model.layer1,
      model.layer2,
      model.layer3,
      model.layer4,
    )

  return model, stages


class Backbone(nn.Module):

  def __init__(
    self,
    model_name,
    weights='imagenet',
    channels=3,
    mode='fix',
    dilated=False,
    strides=None,
    trainable_stem=True,
    trainable_stage4=True,
    trainable_backbone=True,
    backbone_kwargs={},
  ):
    super().__init__()

    self.mode = mode
    self.trainable_stem = trainable_stem
    self.trainable_stage4 = trainable_stage4
    self.trainable_backbone = trainable_backbone
    self.not_training = []
    self.from_scratch_layers = []

    if mode == 'normal':
      self.norm_fn = nn.BatchNorm2d
    elif mode == 'fix':
      self.norm_fn = FixedBatchNorm
    else:
      raise ValueError(f'Unknown mode {mode}. Must be `normal` or `fix`.')

    backbone, stages = build_backbone(
      model_name, dilated, strides, self.norm_fn, weights, channels, **backbone_kwargs,
    )

    self.backbone = backbone
    self.stages = stages

    if not self.trainable_backbone:
      for s in stages:
        set_trainable_layers(s, trainable=False)
      self.not_training.extend(stages)
    else:
      if not self.trainable_stage4:
        self.not_training.extend(stages[:-1])
        for s in stages[:-1]:
          set_trainable_layers(s, trainable=False)

      elif not self.trainable_stem:
        set_trainable_layers(stages[0], trainable=False)
        self.not_training.append(stages[0])

      if self.mode == "fix":
        for s in stages:
          set_trainable_layers(s, torch.nn.BatchNorm2d, trainable=False)
          self.not_training.extend([m for m in s.modules() if isinstance(m, torch.nn.BatchNorm2d)])

  def initialize(self, modules):
    for m in modules:
      if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
      elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm, nn.LayerNorm)):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)

  def get_parameter_groups(self, exclude_partial_names=(), with_names=False):
    names = ([], [], [], [])
    groups = ([], [], [], [])

    scratch_parameters = set()
    all_parameters = set()

    for layer in self.from_scratch_layers:
      for name, param in layer.named_parameters():
        if param in all_parameters:
          continue
        scratch_parameters.add(param)
        all_parameters.add(param)

        if not param.requires_grad:
          continue
        for p in exclude_partial_names:
          if p in name:
            continue

        idx = 2 if "weight" in name else 3
        names[idx].append(name)
        groups[idx].append(param)

    for name, param in self.named_parameters():
      if param in all_parameters:
        continue
      all_parameters.add(param)

      if not param.requires_grad or param in scratch_parameters:
        continue
      for p in exclude_partial_names:
        if p in name:
          continue

      idx = 0 if "weight" in name else 1
      names[idx].append(name)
      groups[idx].append(param)

    if with_names:
      return groups, names

    return groups

  def train(self, mode=True):
    super().train(mode)
    for m in self.not_training:
      m.eval()
    return self


class Classifier(Backbone):

  def __init__(
    self,
    model_name,
    num_classes=20,
    backbone_weights="imagenet",
    channels=3,
    mode='fix',
    dilated=False,
    strides=None,
    trainable_stem=True,
    trainable_stage4=True,
    trainable_backbone=True,
    **backbone_kwargs,
  ):
    super().__init__(
      model_name,
      channels=channels,
      weights=backbone_weights,
      mode=mode,
      dilated=dilated,
      strides=strides,
      trainable_stem=trainable_stem,
      trainable_stage4=trainable_stage4,
      trainable_backbone=trainable_backbone,
      backbone_kwargs=backbone_kwargs,
    )

    self.num_classes = num_classes

    cin = self.backbone.outplanes
    self.classifier = nn.Conv2d(cin, num_classes, 1, bias=False)

    self.from_scratch_layers.extend([self.classifier])
    self.initialize([self.classifier])

  def forward(self, x, with_cam=False):
    outs = self.backbone(x)
    x = outs[-1] if isinstance(outs, tuple) else outs

    if with_cam:
      features = self.classifier(x)
      logits = gap2d(features)
      return logits, features
    else:
      x = gap2d(x, keepdims=True)
      logits = self.classifier(x).view(-1, self.num_classes)
      return logits


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class HPAClassifier(Backbone):

  def __init__(
    self,
    model_name,
    num_classes=20,
    backbone_weights="imagenet",
    channels=3,
    mode='fix',
    dilated=False,
    strides=None,
    trainable_stem=True,
    trainable_stage4=True,
    trainable_backbone=True,
    **backbone_kwargs,
  ):
    super().__init__(
      model_name,
      channels=channels,
      weights=backbone_weights,
      mode=mode,
      dilated=dilated,
      strides=strides,
      trainable_stem=trainable_stem,
      trainable_stage4=trainable_stage4,
      trainable_backbone=trainable_backbone,
      backbone_kwargs=backbone_kwargs,
    )

    self.num_classes = num_classes

    cin = self.backbone.outplanes
    self.classifier = nn.Conv2d(cin, num_classes, 1, bias=False)

    self.from_scratch_layers.extend([self.classifier])
    self.initialize([self.classifier])

    self.pool = GeM()
    self.flatten = nn.Flatten()
    self.dropout = nn.Dropout(p=0.5)

    self.last_linear_cell = nn.Linear(
      in_features=cin, 
      out_features=num_classes)
    self.last_linear_image = nn.Linear(
      in_features=cin, 
      out_features=num_classes)

  def forward(self, x, cnt=16, with_cam=False, cell_logits_to_image_logits=False):
    if with_cam:
      raise NotImplementedError(
        "CAM not currently supported in multi-view mode")

    outs = self.backbone(x)
    features = outs[-1] if isinstance(outs, tuple) else outs

    pooled = self.flatten(self.pool(features))

    if cell_logits_to_image_logits:
      cell_logits = self.last_linear_cell(pooled)
      cell_logits_split = torch.split(cell_logits, cnt.tolist())
      image_logits = torch.stack([p.max(0).values for p in cell_logits_split])

      return cell_logits, image_logits

    pooled_split = torch.split(pooled, cnt.tolist())
    pooled_per_img = torch.stack([p.max(0)[0] for p in pooled_split])

    cell_logits = self.last_linear_cell(pooled)
    image_logits = self.last_linear_image(pooled_per_img)

    return cell_logits, image_logits


class NegativeClassifier(Backbone):
  def __init__(
    self,
    model_name,
    num_classes=1,
    backbone_weights="imagenet",
    channels=3,
    mode='fix',
    dilated=False,
    strides=None,
    trainable_stem=True,
    trainable_stage4=True,
    trainable_backbone=True,
    **backbone_kwargs,
  ):
    super().__init__(
      model_name,
      channels=channels,
      weights=backbone_weights,
      mode=mode,
      dilated=dilated,
      strides=strides,
      trainable_stem=trainable_stem,
      trainable_stage4=trainable_stage4,
      trainable_backbone=trainable_backbone,
      backbone_kwargs=backbone_kwargs,
    )

    self.num_classes = num_classes

    cin = self.backbone.outplanes
    self.classifier = nn.Conv2d(cin, self.num_classes, 1, bias=False)

    self.from_scratch_layers.extend([self.classifier])
    self.initialize([self.classifier])

    self.flatten = nn.Flatten()
    self.dropout = nn.Dropout(p=0.5)

  def forward(self, x):
    # print(f"NegativeClassifier x: {x.shape}")
    features = self.backbone(x)  # Shape: (B, C, H, W)
    features = features[-1] if isinstance(features, tuple) else features
    # print(f"NegativeClassifier features: {features.shape}")
    pooled = F.adaptive_avg_pool2d(features, (1, 1))
    # print(f"NegativeClassifier pooled: {pooled.shape}")
    logits = self.classifier(pooled)
    # print(f"NegativeClassifier logits: {logits.shape}")
    logits = logits.view(logits.size(0), -1)
    # print(f"NegativeClassifier logits after view: {logits.shape}")
    logits = self.dropout(logits)
    # print(f"NegativeClassifier logits after dropout: {logits.shape}")
    return logits

