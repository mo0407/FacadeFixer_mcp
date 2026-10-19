import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from facade_mcp import source_mask_worker as worker


@pytest.fixture
def source_case(tmp_path):
    repo = tmp_path / "source"
    for weights, source in worker._PROFILES.values():
        for relative in (weights, source):
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"local fixture")
    architecture = repo / "training/crack_segmentation/unet/unet_transfer.py"
    architecture.parent.mkdir(parents=True)
    architecture.write_text("# local architecture fixture\n", encoding="utf-8")
    image = tmp_path / "stored.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (8, 6), (170, 150, 130)).save(image, exif=exif)
    return repo, image, tmp_path / "job"


def defect(identifier="d1", code="C", box=None):
    return {"instance_id": identifier, "class_code": code, "bbox_xyxy": box or [1, 1, 5, 4]}


def test_sam_instances_are_independent_full_resolution_and_bbox_clipped(source_case, monkeypatch):
    repo, image, output = source_case
    calls = []

    class Model:
        def predict(self, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(masks=SimpleNamespace(data=np.ones((1, 3, 4))),
                                    boxes=SimpleNamespace(conf=np.asarray([.82])))]

    monkeypatch.setattr(worker, "_make_sam", lambda weights: (Model(), "test"))
    result = worker.run_segmentation(repo, "mobile_sam", image, output,
                                     [defect(), defect("d2", box=[5, 2, 8, 6])], "cpu")
    assert result["status"] == "completed"
    assert len(calls) == 2 and calls[0]["source"].size == (8, 6)
    assert calls[0]["source"].getexif().get(274) is None
    expected = [12, 12]
    for item, area in zip(result["instances"], expected):
        with Image.open(item["mask_path"]) as mask:
            assert mask.size == (8, 6)
            pixels = np.asarray(mask)
            assert set(np.unique(pixels)) <= {0, 255}
            assert (pixels > 0).sum() == area
        raw = np.asarray(Image.open(item["raw_mask_path"]))
        assert raw.shape == (6, 8) and (raw > 0).sum() == 48
        assert item["bbox_clipped_pixels"] == 36
        assert "bbox_clipped_pixels:36" in item["quality_flags"]
    assert np.asarray(Image.open(result["instances"][0]["mask_path"]))[4:, :].sum() == 0
    provenance = result["model_provenance"]
    assert provenance["weights_sha256"] == hashlib.sha256(b"local fixture").hexdigest()
    assert provenance["source_image_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert provenance["source_image_unchanged"] is True and provenance["weights_unchanged"] is True
    assert provenance["source_files_sha256"]
    assert provenance["automatic_weight_download"] is False
    import os
    assert Path(os.environ["MPLCONFIGDIR"]).is_relative_to(output)


def test_empty_model_masks_remain_empty_and_are_flagged(source_case, monkeypatch):
    repo, image, output = source_case
    model = SimpleNamespace(predict=lambda **kwargs: [SimpleNamespace(masks=None)])
    monkeypatch.setattr(worker, "_make_sam", lambda weights: (model, "test"))
    result = worker.run_segmentation(repo, "sam3", image, output, [defect()], "cpu")
    item = result["instances"][0]
    assert "empty_mask" in item["quality_flags"] and item["score"] is None
    assert not np.asarray(Image.open(item["mask_path"])).any()
    assert not np.asarray(Image.open(item["raw_mask_path"])).any()
    assert item["bbox_clipped_pixels"] == 0


def test_point_prompts_are_one_object_and_not_broadcast(source_case, monkeypatch):
    repo, image, output = source_case
    calls = []
    model = SimpleNamespace(predict=lambda **kwargs: calls.append(kwargs) or [])
    monkeypatch.setattr(worker, "_make_sam", lambda weights: (model, "test"))
    worker.run_segmentation(repo, "mobile_sam", image, output, [defect()], "cpu", [[2, 2]], [[3, 3]])
    assert calls[0]["points"] == [[[2., 2.], [3., 3.]]]
    assert calls[0]["labels"] == [[1, 0]]
    with pytest.raises(ValueError, match="exactly one"):
        worker.run_segmentation(repo, "mobile_sam", image, output, [defect(), defect("d2")], "cpu", [[2, 2]])


@pytest.mark.parametrize("item", [defect(box=[-1, 0, 5, 4]), defect(box=[0, 0, 9, 4]),
                                  defect(box=[1.5, 1, 5, 4]), defect(box=[1, 1, 1, 4])])
def test_invalid_boxes_fail_before_model_loading(source_case, monkeypatch, item):
    repo, image, output = source_case
    monkeypatch.setattr(worker, "_make_sam", lambda weights: pytest.fail("should not load model"))
    with pytest.raises(ValueError, match="bbox_xyxy"):
        worker.run_segmentation(repo, "mobile_sam", image, output, [item], "cpu")
    assert not output.exists()


def test_unet_rejects_noncracks_and_clicks_before_model_loading(source_case, monkeypatch):
    repo, image, output = source_case
    monkeypatch.setattr(worker, "_load_unet", lambda *args: pytest.fail("should not load model"))
    with pytest.raises(ValueError, match="only crack"):
        worker.run_segmentation(repo, "unet_crack", image, output, [defect(code="P")], "cpu")
    with pytest.raises(ValueError, match="interactive"):
        worker.run_segmentation(repo, "unet_crack", image, output, [defect()], "cpu", [[2, 2]])


def test_unet_crop_masks_are_placed_in_original_image(source_case, monkeypatch):
    repo, image, output = source_case
    monkeypatch.setattr(worker, "_load_unet", lambda *args: (object(), (448, 448), "test"))

    def predict(model, size, raster, box, device):
        mask = np.ones((raster.height, raster.width), dtype=bool)
        return mask

    monkeypatch.setattr(worker, "_unet_mask", predict)
    result = worker.run_segmentation(repo, "unet_crack", image, output, [defect(code="M")], "cpu")
    pixels = np.asarray(Image.open(result["instances"][0]["mask_path"]))
    assert (pixels > 0).sum() == 12
    assert "training/crack_segmentation/unet/unet_transfer.py" in result["model_provenance"]["source_files_sha256"]


def test_unsupported_profile_and_missing_checkpoint_never_download(source_case, monkeypatch):
    repo, image, output = source_case
    monkeypatch.setattr(worker, "_make_sam", lambda weights: pytest.fail("should not load model"))
    with pytest.raises(ValueError, match="COCO"):
        worker.run_segmentation(repo, "yolo_seg", image, output, [defect()], "cpu")
    (repo / "model_weights/mobile_sam.pt").unlink()
    with pytest.raises(FileNotFoundError):
        worker.run_segmentation(repo, "mobile_sam", image, output, [defect()], "cpu")


def test_missing_scores_do_not_claim_accuracy_and_best_candidate_selected(source_case, monkeypatch):
    repo, image, output = source_case
    arrays = np.zeros((2, 6, 8), dtype=np.float32)
    arrays[0, 1, 1] = 1
    arrays[1, 1:4, 1:5] = 1
    model = SimpleNamespace(predict=lambda **kwargs: [
        SimpleNamespace(masks=SimpleNamespace(data=arrays), boxes=SimpleNamespace(conf=np.asarray([.3, .9])))])
    monkeypatch.setattr(worker, "_make_sam", lambda weights: (model, "test"))
    result = worker.run_segmentation(repo, "sam3", image, output, [defect()], "cpu")
    item = result["instances"][0]
    assert item["score"] == .9 and "model_score_is_not_accuracy" in item["quality_flags"]
    assert (np.asarray(Image.open(item["mask_path"])) > 0).sum() == 12


@pytest.mark.parametrize("target, message", [("image", "Input image changed"), ("weights", "weights changed")])
def test_changed_input_or_weights_rejects_result(source_case, monkeypatch, target, message):
    repo, image, output = source_case

    class Model:
        def predict(self, **kwargs):
            path = image if target == "image" else repo / "model_weights/mobile_sam.pt"
            path.write_bytes(path.read_bytes() + b"changed while inferring")
            return []

    monkeypatch.setattr(worker, "_make_sam", lambda weights: (Model(), "test"))
    with pytest.raises(ValueError, match=message):
        worker.run_segmentation(repo, "mobile_sam", image, output, [defect()], "cpu")


def test_unet_numeric_gpu_index_uses_torch_cuda_device(source_case, monkeypatch):
    repo, image, output = source_case
    devices = []

    def load(repo, weights, device):
        devices.append(device)
        return object(), (448, 448), "test"

    def predict(model, size, raster, box, device):
        devices.append(device)
        return np.zeros((raster.height, raster.width), dtype=bool)

    monkeypatch.setattr(worker, "_load_unet", load)
    monkeypatch.setattr(worker, "_unet_mask", predict)
    result = worker.run_segmentation(repo, "unet_crack", image, output, [defect()], "0")
    assert devices == ["cuda:0", "cuda:0"]
    assert result["model_provenance"]["device"] == "cuda:0"
    assert result["model_provenance"]["configured_device"] == "0"
