"""ACoL(Adversarial Complementary Learning) 기반 약지도 객체 위치 추정 유틸리티.

Zhang et al., CVPR 2018의 두 병렬 분류기와 동적 feature-map erasing 개념을
ResNet-50 backbone에 맞게 구현한다.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import models


@dataclass
class ACoLOutput:
    """ACoL의 두 분류기 결과와 class activation map을 묶는다."""

    logits_a: Tensor
    logits_b: Tensor
    cam_a: Tensor
    cam_b: Tensor
    fused_cam: Tensor

    @property
    def logits(self) -> Tensor:
        """두 classifier의 평균 logit을 최종 분류 점수로 사용한다."""
        return (self.logits_a + self.logits_b) / 2.0


@dataclass(frozen=True)
class ACoLCamResult:
    """한 이미지의 ACoL branch A/B 및 fused CAM 결과."""

    cam_a: Tensor
    cam_b: Tensor
    fused_cam: Tensor
    predicted_class: int
    confidence: float


def _make_grad_scaler(enabled: bool) -> Any:
    """PyTorch 버전 차이를 고려해 CUDA GradScaler를 만든다."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: torch.device, enabled: bool) -> Any:
    """CUDA에서만 FP16 autocast 문맥을 반환한다."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _normalize_per_map(maps: Tensor) -> Tensor:
    """[B, H, W] CAM을 이미지별 [0, 1] 범위로 정규화한다."""
    minimum = maps.amin(dim=(1, 2), keepdim=True)
    maximum = maps.amax(dim=(1, 2), keepdim=True)
    return (maps - minimum) / (maximum - minimum + 1e-8)


class ACoLResNet50(nn.Module):
    """ResNet-50 기반 ACoL 모델.

    Branch A는 feature map에서 가장 강한 판별 영역을 찾는다. 그 영역을 threshold
    이상으로 erase한 feature map을 Branch B로 전달해 서로 보완적인 영역을 학습한다.
    """

    def __init__(
        self,
        num_classes: int,
        use_pretrained: bool = True,
        erase_threshold: float = 0.60,
        train_layer4: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("ACoL 분류 클래스는 2개 이상이어야 합니다.")
        if not 0.0 < erase_threshold < 1.0:
            raise ValueError("erase_threshold는 0과 1 사이여야 합니다.")

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
        try:
            backbone = models.resnet50(weights=weights)
        except Exception as error:
            raise RuntimeError(
                "ResNet-50 가중치를 불러오지 못했습니다. 인터넷 연결 또는 torchvision 캐시를 "
                "확인한 뒤 재실행하십시오."
            ) from error

        # ResNet의 명시적 layer 속성을 유지해 Grad-CAM에서 layer1~layer4를 그대로 선택합니다.
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.erase_threshold = erase_threshold

        # 4 GB VRAM에서는 초반 backbone을 동결하고 layer4와 두 classifier만 fine-tuning합니다.
        for parameter in self.parameters():
            parameter.requires_grad = False
        if train_layer4:
            for parameter in self.layer4.parameters():
                parameter.requires_grad = True

        feature_dim = backbone.fc.in_features
        self.fc = nn.Linear(feature_dim, num_classes)  # Branch A classifier; 기존 CAM 인터페이스 호환
        self.fc_b = nn.Linear(feature_dim, num_classes)  # Erased feature를 받는 Branch B classifier

    def forward_features(self, images: Tensor) -> Tensor:
        """ResNet backbone을 통과시켜 마지막 convolution feature map [B,C,H,W]를 반환한다."""
        features = self.conv1(images)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.maxpool(features)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        return features

    @staticmethod
    def _global_average_pool(features: Tensor) -> Tensor:
        """[B,C,H,W] feature map을 GAP로 [B,C] 벡터로 바꾼다."""
        return torch.flatten(torch.mean(features, dim=(2, 3)), start_dim=1)

    def _class_map(self, features: Tensor, class_indices: Tensor) -> Tensor:
        """Branch A의 예측 클래스별 CAM을 [B,H,W]로 계산한다."""
        class_weights = self.fc.weight[class_indices]  # [B,C]
        return torch.einsum("bc,bchw->bhw", class_weights, features)

    def forward(self, images: Tensor, return_branches: bool = False) -> Tensor | ACoLOutput:
        """두 branch logits와 CAM을 계산하고, 기본적으로 평균 logit을 반환한다."""
        features_a = self.forward_features(images)
        logits_a = self.fc(self._global_average_pool(features_a))

        # Branch A가 가장 강하게 반응한 예측 클래스 CAM에서 top 영역을 찾아 erase합니다.
        # 이 CAM은 오직 동적 erasing mask를 만들기 위한 중간 결과입니다.
        erase_class = logits_a.detach().argmax(dim=1)
        erase_cam = self._class_map(features_a, erase_class)
        normalized_erase_cam = _normalize_per_map(erase_cam)
        erase_mask = (normalized_erase_cam < self.erase_threshold).unsqueeze(1).to(features_a.dtype)
        features_b = features_a * erase_mask

        logits_b = self.fc_b(self._global_average_pool(features_b))
        # 최종 분류 클래스 하나에 대해 A/B branch의 CAM을 각각 만들고 max로 결합합니다.
        # 서로 다른 클래스를 섞지 않으므로 fused map을 클래스별 localization 근거로 해석할 수 있습니다.
        final_class = ((logits_a + logits_b) / 2.0).detach().argmax(dim=1)
        cam_a = self._class_map(features_a, final_class)
        cam_b = self._class_map(features_b, final_class)
        fused_cam = torch.maximum(_normalize_per_map(cam_a), _normalize_per_map(cam_b))

        output = ACoLOutput(
            logits_a=logits_a,
            logits_b=logits_b,
            cam_a=cam_a,
            cam_b=cam_b,
            fused_cam=fused_cam,
        )
        return output if return_branches else output.logits


def create_acol_resnet50(
    num_classes: int,
    device: torch.device,
    use_pretrained: bool = True,
    erase_threshold: float = 0.60,
    train_layer4: bool = True,
) -> ACoLResNet50:
    """현재 디바이스에 맞춘 4 GB VRAM 친화적 ACoL ResNet-50을 생성한다."""
    model = ACoLResNet50(
        num_classes=num_classes,
        use_pretrained=use_pretrained,
        erase_threshold=erase_threshold,
        train_layer4=train_layer4,
    )
    return model.to(device)


def acol_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """optimizer에 전달할 학습 가능 parameter를 반환한다."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("ACoL에 학습 가능 parameter가 없습니다.")
    return parameters


def _run_acol_epoch(
    model: ACoLResNet50,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
    scaler: Any | None = None,
    accumulation_steps: int = 1,
) -> dict[str, float]:
    """Branch A/B cross-entropy 합으로 한 epoch을 학습 또는 검증한다."""
    is_training = optimizer is not None
    if is_training:
        model.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_correct_a = 0
    total_correct_b = 0
    total_samples = 0
    amp_enabled = scaler is not None and scaler.is_enabled()

    for step, (images, labels, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        grad_context = torch.enable_grad() if is_training else torch.inference_mode()

        with grad_context, _autocast(device, amp_enabled):
            output = model(images, return_branches=True)
            loss_a = criterion(output.logits_a, labels)
            loss_b = criterion(output.logits_b, labels)
            # 두 branch가 모두 정답 클래스를 인식하도록 같은 비중으로 supervision을 적용합니다.
            loss = loss_a + loss_b

        if is_training:
            scaled_loss = loss / accumulation_steps
            if amp_enabled:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            is_last_step = step == len(loader)
            if step % accumulation_steps == 0 or is_last_step:
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        logits = output.logits.detach()
        total_loss += loss.detach().item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_correct_a += (output.logits_a.detach().argmax(dim=1) == labels).sum().item()
        total_correct_b += (output.logits_b.detach().argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    if total_samples == 0:
        raise RuntimeError("ACoL DataLoader가 비어 있습니다.")
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "accuracy_a": total_correct_a / total_samples,
        "accuracy_b": total_correct_b / total_samples,
    }


def train_acol_model(
    model: ACoLResNet50,
    train_loader: DataLoader[Any],
    valid_loader: DataLoader[Any],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    accumulation_steps: int,
    checkpoint_dir: Path,
    use_amp: bool,
) -> list[dict[str, float]]:
    """ACoL을 end-to-end로 학습하고 최고 검증 정확도 checkpoint를 저장한다."""
    if epochs < 1:
        raise ValueError("epochs는 1 이상이어야 합니다.")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(acol_trainable_parameters(model), lr=learning_rate, weight_decay=1e-4)
    scaler = _make_grad_scaler(enabled=use_amp and device.type == "cuda")

    history: list[dict[str, float]] = []
    best_accuracy = float("-inf")
    for epoch in range(1, epochs + 1):
        train = _run_acol_epoch(
            model, train_loader, criterion, device, optimizer, scaler, accumulation_steps
        )
        valid = _run_acol_epoch(model, valid_loader, criterion, device)
        record = {
            "epoch": float(epoch),
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "valid_loss": valid["loss"],
            "valid_accuracy": valid["accuracy"],
            "valid_accuracy_a": valid["accuracy_a"],
            "valid_accuracy_b": valid["accuracy_b"],
        }
        history.append(record)
        print(
            f"ACoL epoch {epoch}/{epochs} | train loss={record['train_loss']:.4f}, "
            f"acc={record['train_accuracy']:.2%} | valid loss={record['valid_loss']:.4f}, "
            f"acc={record['valid_accuracy']:.2%}"
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": record,
            "erase_threshold": model.erase_threshold,
        }
        torch.save(state, checkpoint_dir / "latest.pt")
        if record["valid_accuracy"] > best_accuracy:
            best_accuracy = record["valid_accuracy"]
            torch.save(state, checkpoint_dir / "best.pt")
    return history


def load_acol_checkpoint(
    model: ACoLResNet50,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """ACoL checkpoint를 현재 device에 맞게 불러온다."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError("ACoL checkpoint에 model_state_dict가 없습니다.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return checkpoint


def generate_acol_cam(
    model: ACoLResNet50,
    image: Tensor,
) -> ACoLCamResult:
    """실제 입력 이미지에서 branch A/B와 fused ACoL CAM을 생성한다."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("ACoL CAM 입력은 shape [1, C, H, W]여야 합니다.")

    model.eval()
    with torch.inference_mode():
        output = model(image, return_branches=True)
        logits = output.logits
        predicted_class = int(logits.argmax(dim=1).item())
        confidence = float(torch.softmax(logits, dim=1)[0, predicted_class].item())

    return ACoLCamResult(
        cam_a=_normalize_per_map(output.cam_a).squeeze(0).detach().cpu(),
        cam_b=_normalize_per_map(output.cam_b).squeeze(0).detach().cpu(),
        fused_cam=output.fused_cam.squeeze(0).detach().cpu(),
        predicted_class=predicted_class,
        confidence=confidence,
    )
