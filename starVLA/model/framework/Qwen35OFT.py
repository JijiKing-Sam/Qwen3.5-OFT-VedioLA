from __future__ import annotations

from starVLA.model.framework.QwenOFT import Qwenvl_OFT
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("Qwen35OFT")
class Qwen35OFT(Qwenvl_OFT):
    """
    OFT head on top of the official Qwen3.5 multimodal backbone.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.qwen35_interface = self.qwen_vl_interface
