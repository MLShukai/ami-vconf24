import pytest
import torch

from ami.models.model_wrapper import (
    ModelWrapper,
    ThreadSafeInferenceWrapper,
    default_infer,
)
from tests.helpers import ModelMultiplyP, skip_if_gpu_is_not_available


@skip_if_gpu_is_not_available()
def test_default_infer(gpu_device: torch.device):
    wrapper = ModelWrapper(ModelMultiplyP(), gpu_device, True)
    wrapper.to_default_device()
    data = torch.randn(10)
    assert data.device == torch.device("cpu")
    assert default_infer(wrapper, data).device == gpu_device


class TestWrappers:
    def test_device_cpu(self):
        device = torch.device("cpu")
        m = ModelWrapper(ModelMultiplyP(), device, True)

        assert m.device == device
        m.to_default_device()
        assert m.device == m.device

    @skip_if_gpu_is_not_available()
    def test_device_gpu(self, gpu_device: torch.device):
        m = ModelWrapper(ModelMultiplyP(), gpu_device, True)

        assert m.device.type == "cpu"
        m.to_default_device()
        assert m.device == gpu_device

    def test_inference_wrapper_property(self):
        m = ModelMultiplyP()
        mw = ModelWrapper(m, "cpu", True)
        inference_wrapper = mw.inference_wrapper
        assert mw.inference_wrapper is inference_wrapper
        assert isinstance(inference_wrapper, ThreadSafeInferenceWrapper)
        assert inference_wrapper.model is not m
        assert inference_wrapper.model.p is not m.p
        assert inference_wrapper.model.p == m.p

        mw = ModelWrapper(m, "cpu", False)
        with pytest.raises(RuntimeError):
            mw.inference_wrapper

    def test_infer_cpu(self):
        mw = ModelWrapper(ModelMultiplyP(), "cpu", True)
        mw.to_default_device()
        inference = mw.inference_wrapper

        data = torch.randn(10)
        assert isinstance(inference(data), torch.Tensor)

    @skip_if_gpu_is_not_available()
    def test_infer_gpu(self, gpu_device: torch.device):
        mw = ModelWrapper(ModelMultiplyP(), gpu_device, True)
        mw.to_default_device()
        inference = mw.inference_wrapper

        data = torch.randn(10)
        out: torch.Tensor = inference(data)
        assert out.device == gpu_device

    def test_freeze_model_and_unfreeze_model(self) -> None:
        mw = ModelWrapper(ModelMultiplyP(), "cpu", True)
        assert mw.model.p.requires_grad is True

        mw.freeze_model()
        assert mw.model.p.requires_grad is False
        assert mw.model.training is False

        mw.unfreeze_model()
        assert mw.model.p.requires_grad is True
        assert mw.model.training is True

    def test_load_parameter_file(self, tmp_path) -> None:
        param_file = tmp_path / "param.pt"

        m = ModelMultiplyP()
        torch.save(m.state_dict(), param_file)

        mw = ModelWrapper(ModelMultiplyP(), "cpu", True, parameter_file=param_file)

        assert torch.equal(mw.model.p, m.p)

    def test_inference_thread_only(self):
        m = ModelWrapper(ModelMultiplyP(), has_inference=True, inference_thread_only=True)
        iw = m.inference_wrapper

        assert m.inference_thread_only
        assert iw._wrapper is m

        # default false.
        m = ModelWrapper(ModelMultiplyP(), has_inference=True)
        iw = m.inference_wrapper
        assert not m.inference_thread_only
        assert iw._wrapper is not m

        # test consistency between `has_inference` and `inference_thread_only`
        with pytest.raises(ValueError):
            ModelWrapper(ModelMultiplyP(), has_inference=False, inference_thread_only=True)
