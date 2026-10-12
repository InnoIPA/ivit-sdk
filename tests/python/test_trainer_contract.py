import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


class SmallDataset:
    num_classes = 2
    class_names = ["暗色", 'bright"label']

    def __init__(self, segmentation=False):
        self.segmentation = segmentation

    def __len__(self):
        return 4

    def __getitem__(self, index):
        image = np.full((16, 16, 3), 50 + index * 40, np.uint8)
        target = (
            {"mask": np.full((16, 16), index % 2, np.uint8)}
            if self.segmentation
            else index % 2
        )
        return image, target


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, image):
        return self.fc(image.mean(dim=(2, 3)))


class TinySegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 1)

    def forward(self, image):
        return {"out": self.conv(image)}


@pytest.mark.parametrize("segmentation", [False, True])
def test_task_trainer_fit_evaluate_export_and_native_reload(tmp_path, segmentation):
    import ivit
    from ivit.train import ClassificationTrainer, SegmentationTrainer, Trainer

    assert Trainer is ClassificationTrainer
    torch.set_num_threads(2)
    torch.manual_seed(7)
    dataset = SmallDataset(segmentation)
    network = TinySegmentor() if segmentation else TinyClassifier()
    trainer_type = SegmentationTrainer if segmentation else ClassificationTrainer
    trainer = trainer_type(
        model=network,
        dataset=dataset,
        val_dataset=dataset,
        device="cpu",
        epochs=1,
        batch_size=2,
        num_workers=0,
        image_size=16,
        freeze_backbone=False,
    )
    before = [param.detach().clone() for param in trainer.model.parameters()]
    history = trainer.fit()
    assert history and np.isfinite(history[-1]["loss"])
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, trainer.model.parameters())
    )
    assert trainer.evaluate()
    path = tmp_path / ("segmentation.onnx" if segmentation else "classification.onnx")
    trainer.export(str(path))
    assert path.exists()
    assert (
        path.with_suffix(".labels.txt").read_text(encoding="utf-8").splitlines()
        == dataset.class_names
    )
    if not ivit.is_cpp_available():
        pytest.skip("Training/export passed; native reload requires bindings")
    model = ivit.load(
        str(path),
        task="segmentation" if segmentation else "classification",
        backend="openvino",
        device="cpu",
        use_cache=False,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    config = ivit.InferConfig()
    config.conf_threshold = 0
    image, _ = dataset[0]
    actual = model.predict(image[:, :, ::-1].copy(), config)
    normalized = (
        image.astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406], np.float32)
    ) / np.array([0.229, 0.224, 0.225], np.float32)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy())[None]
    with torch.no_grad():
        reference = trainer.model(tensor)
    if segmentation:
        np.testing.assert_array_equal(
            actual.segmentation_mask, reference["out"].argmax(dim=1)[0].numpy()
        )
    else:
        expected = torch.softmax(reference, dim=1)[0].numpy()
        scores = {item.class_id: item.score for item in actual.classifications}
        np.testing.assert_allclose([scores[i] for i in range(2)], expected, atol=1e-5)


def test_detection_trainer_validates_supported_model_and_dependency():
    from ivit.train import DetectionTrainer

    with pytest.raises(ValueError, match="Unknown detection model"):
        DetectionTrainer("wrong")
    with pytest.raises(ValueError, match="data_dir"):
        DetectionTrainer("yolox-s")


def test_export_int8_cannot_silently_produce_fp32(tmp_path):
    from ivit.train import ModelExporter

    exporter = ModelExporter(TinyClassifier(), "cpu")
    path = tmp_path / "wrong.onnx"
    with pytest.raises(ValueError, match="calibration"):
        exporter.export(str(path), quantize="int8")
    assert not path.exists()


def test_tensor_rt_export_uses_native_builder_and_retains_labels(tmp_path):
    import ivit
    from ivit.train import ModelExporter

    if not any(device.backend == "tensorrt" for device in ivit.list_devices()):
        pytest.skip("TensorRT GPU required")
    network = TinyClassifier().eval()
    path = tmp_path / "export.engine"
    classes = ["first", "second"]
    result = ModelExporter(network, "cpu").export(
        str(path), format="tensorrt", input_shape=(1, 3, 16, 16), class_names=classes
    )
    assert result == str(path)
    assert path.with_suffix(".labels.txt").read_text().splitlines() == classes
    model = ivit.load(
        result,
        backend="tensorrt",
        device="cuda:0",
        task="classification",
        use_cache=False,
    )
    options = ivit.InferConfig()
    options.conf_threshold = 0
    actual = model.predict(np.zeros((16, 16, 3), np.uint8), options)
    with torch.no_grad():
        expected = torch.softmax(network(torch.zeros(1, 3, 16, 16)), dim=1)[0].numpy()
    scores = {item.class_id: item.score for item in actual.classifications}
    np.testing.assert_allclose([scores[i] for i in range(2)], expected, atol=1e-5)
