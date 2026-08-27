"""Stanford Dogs 기반 CAM·Grad-CAM 실험을 위한 재사용 모듈."""

from __future__ import annotations

import xml.etree.ElementTree as element_tree
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import scipy.io as sio
import torch
from PIL import Image, ImageFile
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DogRecord:
    """원본 이미지 상대 경로와 0-base 클래스 인덱스."""

    relative_path: Path
    target: int


@dataclass(frozen=True)
class CamResult:
    """한 이미지에 대한 CAM 또는 Grad-CAM 계산 결과."""

    heatmap: np.ndarray
    predicted_class: int
    confidence: float


def _unwrap_matlab_value(value: Any) -> str:
    """MATLAB cell array 원소를 안전하게 UTF-8 문자열로 변환한다."""
    while isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"예상과 다른 MATLAB 배열 크기: {value.shape}")
        value = value.reshape(-1)[0]

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if not isinstance(value, str):
        raise TypeError(f"파일 경로는 문자열이어야 합니다. 실제 타입: {type(value)!r}")
    return value


def load_dog_records(split_mat_path: Path) -> tuple[list[DogRecord], list[str]]:
    """공식 Stanford Dogs split .mat 파일을 0-base 레코드로 변환한다."""
    if not split_mat_path.is_file():
        raise FileNotFoundError(f"분할 목록 파일을 찾지 못했습니다: {split_mat_path}")

    split = sio.loadmat(split_mat_path)
    if "file_list" not in split or "labels" not in split:
        raise KeyError(".mat 파일에는 'file_list'와 'labels' 키가 필요합니다.")

    file_list = np.asarray(split["file_list"]).reshape(-1)
    labels = np.asarray(split["labels"]).reshape(-1)
    if len(file_list) != len(labels):
        raise ValueError("file_list와 labels의 길이가 다릅니다.")

    label_to_class: dict[int, str] = {}
    raw_records: list[tuple[Path, int]] = []
    for raw_path, raw_label in zip(file_list, labels, strict=True):
        relative_path = Path(_unwrap_matlab_value(raw_path))
        if len(relative_path.parts) < 2:
            raise ValueError(f"클래스 폴더가 포함되지 않은 경로입니다: {relative_path}")

        raw_target = int(raw_label)
        label_to_class.setdefault(raw_target, relative_path.parts[0])
        if label_to_class[raw_target] != relative_path.parts[0]:
            raise ValueError("동일 레이블이 서로 다른 클래스 폴더에 연결되었습니다.")
        raw_records.append((relative_path, raw_target))

    ordered_labels = sorted(label_to_class)
    expected_labels = list(range(1, len(ordered_labels) + 1))
    if ordered_labels != expected_labels:
        raise ValueError("Stanford Dogs 레이블은 1부터 연속된 정수여야 합니다.")

    classes = [label_to_class[label] for label in ordered_labels]
    records = [DogRecord(path, label - 1) for path, label in raw_records]
    return records, classes


def parse_annotation(annotation_path: Path) -> tuple[float, float, float, float]:
    """Stanford Dogs XML annotation의 첫 번째 bounding box를 읽는다."""
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation 파일을 찾지 못했습니다: {annotation_path}")

    root = element_tree.parse(annotation_path).getroot()
    bndbox = root.find("./object/bndbox")
    if bndbox is None:
        raise ValueError(f"bndbox 태그를 찾지 못했습니다: {annotation_path}")

    keys = ("xmin", "ymin", "xmax", "ymax")
    values: list[float] = []
    for key in keys:
        value = bndbox.findtext(key)
        if value is None:
            raise ValueError(f"{key} 태그를 찾지 못했습니다: {annotation_path}")
        values.append(float(value))
    return tuple(values)  # type: ignore[return-value]


def scaled_bbox(
    bbox: Sequence[float],
    original_size: tuple[int, int],
    output_size: int,
) -> Tensor:
    """원본 좌표계 bbox를 정사각형 모델 입력 좌표계로 스케일한다."""
    original_width, original_height = original_size
    if original_width <= 0 or original_height <= 0:
        raise ValueError(f"유효하지 않은 원본 크기: {original_size}")

    xmin, ymin, xmax, ymax = bbox
    scale_x = output_size / original_width
    scale_y = output_size / original_height
    return torch.tensor(
        [xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y],
        dtype=torch.float32,
    )


class StanfordDogsDataset(Dataset[tuple[Tensor, int, Tensor]]):
    """원본 파일 구조를 복사하지 않고 이미지·레이블·bbox를 제공하는 데이터셋."""

    def __init__(
        self,
        image_root: Path,
        annotation_root: Path,
        split_mat_path: Path,
        image_size: int = 224,
    ) -> None:
        self.image_root = Path(image_root)
        self.annotation_root = Path(annotation_root)
        self.image_size = image_size
        self.records, self.classes = load_dog_records(Path(split_mat_path))
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Images 폴더를 찾지 못했습니다: {self.image_root}")
        if not self.annotation_root.is_dir():
            raise FileNotFoundError(f"Annotation 폴더를 찾지 못했습니다: {self.annotation_root}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int, Tensor]:
        record = self.records[index]
        image_path = self.image_root / record.relative_path
        annotation_path = self.annotation_root / record.relative_path.with_suffix("")
        if not image_path.is_file():
            raise FileNotFoundError(f"이미지를 찾지 못했습니다: {image_path}")

        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")
            original_size = image.size
            tensor = self.transform(image)

        bbox = scaled_bbox(
            parse_annotation(annotation_path),
            original_size=original_size,
            output_size=self.image_size,
        )
        return tensor, record.target, bbox


def make_datasets(
    image_root: Path,
    annotation_root: Path,
    lists_root: Path,
    image_size: int = 224,
) -> tuple[StanfordDogsDataset, StanfordDogsDataset]:
    """공식 train/test 목록을 사용해 데이터셋을 생성한다."""
    train_dataset = StanfordDogsDataset(
        image_root=image_root,
        annotation_root=annotation_root,
        split_mat_path=Path(lists_root) / "train_list.mat",
        image_size=image_size,
    )
    valid_dataset = StanfordDogsDataset(
        image_root=image_root,
        annotation_root=annotation_root,
        split_mat_path=Path(lists_root) / "test_list.mat",
        image_size=image_size,
    )
    if train_dataset.classes != valid_dataset.classes:
        raise ValueError("train/test 클래스 순서가 일치하지 않습니다.")
    return train_dataset, valid_dataset


def limit_dataset(dataset: Dataset[Any], max_samples: int | None) -> Dataset[Any]:
    """재현 가능하게 선두 샘플만 선택한다. None이면 전체를 사용한다."""
    if max_samples is None:
        return dataset
    if max_samples <= 0:
        raise ValueError("max_samples는 양의 정수 또는 None이어야 합니다.")
    return Subset(dataset, range(min(max_samples, len(dataset))))


def make_loaders(
    train_dataset: Dataset[Any],
    valid_dataset: Dataset[Any],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Windows Notebook 안정성을 고려한 DataLoader를 생성한다."""
    if batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")
    if num_workers < 0:
        raise ValueError("num_workers는 0 이상이어야 합니다.")

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": False,
    }
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(valid_dataset, shuffle=False, **common),
    )


def create_resnet50(
    num_classes: int,
    device: torch.device,
    use_pretrained: bool = True,
    train_layer4: bool = True,
) -> nn.Module:
    """4 GB VRAM을 고려해 layer4와 분류기만 미세조정하는 ResNet-50을 만든다."""
    if num_classes < 2:
        raise ValueError("분류 클래스는 2개 이상이어야 합니다.")

    weights = models.ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
    try:
        model = models.resnet50(weights=weights)
    except Exception as error:
        raise RuntimeError(
            "ResNet-50 가중치를 불러오지 못했습니다. 인터넷 연결 또는 torchvision 캐시를 "
            "확인한 뒤 재실행하십시오. 오프라인 실험이면 USE_PRETRAINED=False로 변경하십시오."
        ) from error

    for parameter in model.parameters():
        parameter.requires_grad = False
    if train_layer4:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """optimizer에 전달할 학습 가능 파라미터를 반환한다."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("학습 가능 파라미터가 없습니다.")
    return parameters


def _make_grad_scaler(enabled: bool) -> Any:
    """PyTorch 버전 차이를 고려해 CUDA GradScaler를 생성한다."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: torch.device, enabled: bool) -> Any:
    """CUDA일 때만 FP16 autocast 컨텍스트를 반환한다."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
    scaler: Any | None = None,
    accumulation_steps: int = 1,
) -> tuple[float, float]:
    """한 epoch의 학습 또는 검증을 실행하고 loss·accuracy를 반환한다."""
    is_training = optimizer is not None
    if is_training:
        model.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    amp_enabled = scaler is not None and scaler.is_enabled()

    for step, (images, labels, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        grad_context = torch.enable_grad() if is_training else torch.inference_mode()

        try:
            with grad_context, _autocast(device, amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA OOM이 발생했습니다. BATCH_SIZE=1을 유지한 채 IMAGE_SIZE 또는 "
                    "학습 범위(layer4)를 먼저 줄이십시오."
                ) from error
            raise

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

        total_loss += loss.detach().item() * labels.size(0)
        total_correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    if total_samples == 0:
        raise RuntimeError("DataLoader가 비어 있습니다.")
    return total_loss / total_samples, total_correct / total_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    valid_loader: DataLoader[Any],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    accumulation_steps: int,
    checkpoint_dir: Path,
    use_amp: bool,
) -> list[dict[str, float]]:
    """메모리 보수적 설정으로 학습하고 epoch별 최신·최고 checkpoint를 저장한다."""
    if epochs < 1:
        raise ValueError("epochs는 1 이상이어야 합니다.")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=learning_rate, weight_decay=1e-4)
    scaler = _make_grad_scaler(enabled=use_amp and device.type == "cuda")

    history: list[dict[str, float]] = []
    best_accuracy = float("-inf")
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            accumulation_steps=accumulation_steps,
        )
        valid_loss, valid_accuracy = _run_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
        )
        record = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_accuracy,
        }
        history.append(record)
        print(
            f"Epoch {epoch}/{epochs} | train loss={train_loss:.4f}, acc={train_accuracy:.2%} | "
            f"valid loss={valid_loss:.4f}, acc={valid_accuracy:.2%}"
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": record,
        }
        torch.save(state, checkpoint_dir / "latest.pt")
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(state, checkpoint_dir / "best.pt")
    return history


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    """state_dict 형식 checkpoint를 안전하게 현재 device로 불러온다."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint에 model_state_dict가 없습니다.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return checkpoint


def _validate_image_tensor(image: Tensor) -> None:
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("CAM 입력은 shape [1, C, H, W]의 단일 이미지 배치여야 합니다.")


def _normalize_heatmap(heatmap: Tensor) -> np.ndarray:
    heatmap = torch.relu(heatmap)
    minimum = heatmap.min()
    maximum = heatmap.max()
    normalized = (heatmap - minimum) / (maximum - minimum + 1e-8)
    return normalized.detach().float().cpu().numpy()


def generate_cam(model: nn.Module, image: Tensor, target_class: int | None = None) -> CamResult:
    """ResNet의 layer4 feature map과 fc 가중치로 CAM을 생성한다."""
    _validate_image_tensor(image)
    if not hasattr(model, "layer4") or not hasattr(model, "fc"):
        raise TypeError("CAM은 layer4와 fc를 가진 ResNet 계열 모델에서 사용하십시오.")

    activations: dict[str, Tensor] = {}

    def forward_hook(_: nn.Module, __: tuple[Tensor, ...], output: Tensor) -> None:
        activations["value"] = output.detach()

    handle = model.layer4.register_forward_hook(forward_hook)
    try:
        model.eval()
        with torch.inference_mode():
            logits = model(image)
    finally:
        handle.remove()

    if "value" not in activations:
        raise RuntimeError("CAM feature map을 수집하지 못했습니다.")
    predicted = int(logits.argmax(dim=1).item())
    class_index = predicted if target_class is None else target_class
    if not 0 <= class_index < logits.shape[1]:
        raise ValueError(f"target_class 범위 오류: {class_index}")

    feature_map = activations["value"][0]
    weights = model.fc.weight[class_index].detach().to(feature_map.device)
    heatmap = torch.einsum("c,chw->hw", weights, feature_map)
    confidence = float(torch.softmax(logits, dim=1)[0, class_index].item())
    return CamResult(_normalize_heatmap(heatmap), predicted, confidence)


def generate_grad_cam(
    model: nn.Module,
    image: Tensor,
    target_layer: nn.Module,
    target_class: int | None = None,
) -> CamResult:
    """선택한 convolution layer에서 Grad-CAM을 생성한다."""
    _validate_image_tensor(image)
    activations: dict[str, Tensor] = {}

    def forward_hook(_: nn.Module, __: tuple[Tensor, ...], output: Tensor) -> None:
        output.retain_grad()
        activations["value"] = output

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(image)
        predicted = int(logits.argmax(dim=1).item())
        class_index = predicted if target_class is None else target_class
        if not 0 <= class_index < logits.shape[1]:
            raise ValueError(f"target_class 범위 오류: {class_index}")
        logits[0, class_index].backward()
    finally:
        handle.remove()

    if "value" not in activations or activations["value"].grad is None:
        raise RuntimeError("Grad-CAM의 activation 또는 gradient를 수집하지 못했습니다.")

    feature_map = activations["value"][0]
    gradients = activations["value"].grad[0]
    weights = gradients.mean(dim=(1, 2))
    heatmap = torch.einsum("c,chw->hw", weights, feature_map)
    confidence = float(torch.softmax(logits.detach(), dim=1)[0, class_index].item())
    return CamResult(_normalize_heatmap(heatmap), predicted, confidence)


def resize_heatmap(heatmap: np.ndarray, width: int, height: int) -> np.ndarray:
    """heatmap을 이미지 표시 크기에 맞게 선형 보간한다."""
    return cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)


def unnormalize_image(image: Tensor) -> np.ndarray:
    """ImageNet 정규화 tensor를 표시 가능한 uint8 RGB 이미지로 복원한다."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("입력 이미지는 shape [3, H, W]여야 합니다.")
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    array = np.clip(array * std + mean, 0.0, 1.0)
    return (array * 255).round().astype(np.uint8)


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """RGB 이미지에 heatmap을 겹쳐 표시한다."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha는 0과 1 사이여야 합니다.")
    heatmap = np.clip(heatmap, 0.0, 1.0)
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1.0 - alpha, colored_rgb, alpha, 0.0)


def extract_bbox(heatmap: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int, int] | None:
    """임계값 이상 활성화 영역 중 가장 큰 연결 요소를 bbox로 반환한다."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold는 0과 1 사이여야 합니다.")
    binary = (heatmap >= threshold).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if component_count <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    component_index = int(np.argmax(areas)) + 1
    x, y, width, height, _ = stats[component_index]
    return int(x), int(y), int(x + width - 1), int(y + height - 1)


def bbox_iou(
    first: Sequence[float] | None,
    second: Sequence[float] | None,
) -> float:
    """두 (xmin, ymin, xmax, ymax) bbox의 IoU를 계산한다."""
    if first is None or second is None:
        return 0.0
    x1_min, y1_min, x1_max, y1_max = map(float, first)
    x2_min, y2_min, x2_max, y2_max = map(float, second)
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    intersection = max(0.0, inter_xmax - inter_xmin) * max(0.0, inter_ymax - inter_ymin)
    area_first = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area_second = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    union = area_first + area_second - intersection
    return 0.0 if union <= 0.0 else intersection / union


def draw_bbox(
    image_rgb: np.ndarray,
    bbox: Sequence[float] | None,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """RGB 이미지에 bbox를 그린 복사본을 반환한다."""
    canvas = image_rgb.copy()
    if bbox is None:
        return canvas
    xmin, ymin, xmax, ymax = (int(round(value)) for value in bbox)
    return cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), color, thickness)


def choose_evenly_spaced_indices(length: int, count: int) -> list[int]:
    """데이터셋 전체에 걸친 재현 가능한 샘플 인덱스를 선택한다."""
    if length <= 0:
        raise ValueError("데이터셋 길이는 양수여야 합니다.")
    if count <= 0:
        raise ValueError("count는 양수여야 합니다.")
    return np.linspace(0, length - 1, num=min(length, count), dtype=int).tolist()


def mean_localization_iou(
    model: nn.Module,
    dataset: Dataset[tuple[Tensor, int, Tensor]],
    device: torch.device,
    target_layer: nn.Module,
    sample_indices: Iterable[int],
    threshold: float = 0.5,
) -> list[dict[str, float | int | bool]]:
    """여러 실제 테스트 샘플에서 Grad-CAM bbox와 ground truth bbox의 IoU를 평가한다."""
    rows: list[dict[str, float | int | bool]] = []
    for index in sample_indices:
        image, target, ground_truth = dataset[index]
        result = generate_grad_cam(model, image.unsqueeze(0).to(device), target_layer=target_layer)
        resized = resize_heatmap(result.heatmap, width=image.shape[2], height=image.shape[1])
        predicted_bbox = extract_bbox(resized, threshold=threshold)
        rows.append(
            {
                "index": int(index),
                "target": int(target),
                "prediction": result.predicted_class,
                "correct_class": result.predicted_class == int(target),
                "iou": bbox_iou(predicted_bbox, ground_truth.tolist()),
            }
        )
    return rows


def memory_smoke_test(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    learning_rate: float,
    use_amp: bool,
) -> dict[str, float]:
    """실제 데이터 한 배치로 forward·backward·optimizer step을 수행한다."""
    try:
        images, labels, _ = next(iter(loader))
    except StopIteration as error:
        raise RuntimeError("memory smoke test를 위한 학습 데이터가 없습니다.") from error

    model.train()
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = _make_grad_scaler(enabled=use_amp and device.type == "cuda")
    images = images.to(device, non_blocking=device.type == "cuda")
    labels = labels.to(device, non_blocking=device.type == "cuda")

    optimizer.zero_grad(set_to_none=True)
    with _autocast(device, scaler.is_enabled()):
        logits = model(images)
        loss = criterion(logits, labels)
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    result = {"smoke_loss": float(loss.detach().cpu().item())}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        result.update(
            {
                "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
                "vram_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
                "vram_free_gb": round(free / 1024**3, 3),
                "vram_total_gb": round(total / 1024**3, 3),
            }
        )
    return result


STANFORD_DOGS_HTTPS_URLS = {
    "images.tar": "https://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
    "annotation.tar": "https://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar",
    "lists.tar": "https://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar",
}
STANFORD_DOGS_HTTP_URLS = {
    archive_name: url.replace("https://", "http://", 1)
    for archive_name, url in STANFORD_DOGS_HTTPS_URLS.items()
}


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    """경로 이탈을 차단하며 tar 아카이브를 추출한다."""
    import tarfile

    destination = destination.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if not member_path.is_relative_to(destination):
                raise RuntimeError(f"안전하지 않은 tar 경로가 감지되었습니다: {member.name}")
        archive.extractall(destination)


def _download_archive(url: str, destination: Path) -> None:
    """HTTP(S) 응답을 임시 파일에 저장한 뒤 원자적으로 archive를 만든다."""
    from shutil import copyfileobj
    from urllib.request import Request, urlopen

    temporary_path = destination.with_suffix(destination.suffix + ".part")
    temporary_path.unlink(missing_ok=True)
    request = Request(url, headers={"User-Agent": "GD01-CAM-Notebook/1.0"})
    try:
        with urlopen(request, timeout=60) as response, temporary_path.open("wb") as file:
            copyfileobj(response, file)
        temporary_path.replace(destination)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise


def prepare_stanford_dogs(
    data_root: Path,
    allow_official_http_fallback: bool = True,
) -> dict[str, Path]:
    """공식 Stanford Dogs 데이터를 다운로드하고 필요한 경로를 반환한다.

    해당 공식 서버는 현재 HTTPS certificate subject 불일치를 반환할 수 있다. 이 경우
    인증서 검증을 해제하지 않고, 사용자가 허용한 경우에만 같은 공식 도메인의 HTTP
    공개 파일 엔드포인트를 명시적으로 대체 사용한다.
    """
    import ssl
    import tarfile
    import warnings
    from urllib.error import URLError

    data_root = Path(data_root)
    raw_dir = data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    expected = {
        "images": data_root / "Images",
        "annotations": data_root / "Annotation",
        "lists": data_root,
    }
    archive_to_expected = {
        "images.tar": expected["images"],
        "annotation.tar": expected["annotations"],
        "lists.tar": data_root / "train_list.mat",
    }

    for archive_name, https_url in STANFORD_DOGS_HTTPS_URLS.items():
        archive_path = raw_dir / archive_name
        required_path = archive_to_expected[archive_name]
        if required_path.exists():
            print(f"재사용: {required_path}")
            continue
        if archive_path.exists() and not tarfile.is_tarfile(archive_path):
            warnings.warn(
                f"불완전한 아카이브를 삭제하고 다시 받습니다: {archive_path}",
                RuntimeWarning,
                stacklevel=2,
            )
            archive_path.unlink()
        if not archive_path.exists():
            print(f"다운로드(HTTPS): {https_url}")
            try:
                _download_archive(https_url, archive_path)
            except (ssl.SSLCertVerificationError, URLError) as error:
                reason = getattr(error, "reason", error)
                if not isinstance(reason, ssl.SSLCertVerificationError):
                    raise RuntimeError(
                        "HTTPS 데이터 다운로드에 실패했습니다. 네트워크를 확인하거나 아카이브를 "
                        f"{raw_dir}에 직접 저장한 뒤 다시 실행하십시오."
                    ) from error
                if not allow_official_http_fallback:
                    raise RuntimeError(
                        "공식 서버의 HTTPS 인증서가 현재 검증되지 않습니다. 인증서 검증을 "
                        "해제하지 않았습니다. raw 폴더에 archive를 직접 저장하거나 "
                        "ALLOW_OFFICIAL_HTTP_FALLBACK=True로 설정하십시오."
                    ) from error
                http_url = STANFORD_DOGS_HTTP_URLS[archive_name]
                warnings.warn(
                    "공식 Stanford 서버의 HTTPS 인증서가 일치하지 않아, 암호화되지 않은 "
                    f"공개 HTTP 엔드포인트로 대체합니다: {http_url}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                print(f"다운로드(공식 HTTP 대체): {http_url}")
                try:
                    _download_archive(http_url, archive_path)
                except OSError as fallback_error:
                    raise RuntimeError(
                        "공식 HTTP 대체 다운로드도 실패했습니다. 네트워크를 확인하거나 "
                        f"아카이브를 {raw_dir}에 직접 저장한 뒤 다시 실행하십시오."
                    ) from fallback_error
            except OSError as error:
                raise RuntimeError(
                    "HTTPS 데이터 다운로드에 실패했습니다. 네트워크를 확인하거나 아카이브를 "
                    f"{raw_dir}에 직접 저장한 뒤 다시 실행하십시오."
                ) from error
        print(f"추출: {archive_path.name}")
        _safe_extract_tar(archive_path, data_root)

    train_list = expected["lists"] / "train_list.mat"
    test_list = expected["lists"] / "test_list.mat"
    if not train_list.is_file() or not test_list.is_file():
        raise FileNotFoundError("train_list.mat 또는 test_list.mat을 찾지 못했습니다.")
    return expected
