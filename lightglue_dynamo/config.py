from enum import auto
from backports.strenum import StrEnum
from typing import Any, Dict


class InferenceDevice(StrEnum):
    cpu = auto()
    cuda = auto()
    tensorrt = auto()
    openvino = auto()


class Extractor(StrEnum):
    superpoint = auto()
    disk = auto()

    @property
    def input_dim_divisor(self) -> int:
        if self == Extractor.superpoint:
            return 8
        elif self == Extractor.disk:
            return 16

    @property
    def input_channels(self) -> int:
        if self == Extractor.superpoint:
            return 1
        elif self == Extractor.disk:
            return 3

    @property
    def lightglue_config(self) -> Dict[str, Any]:
        if self == Extractor.superpoint:
            return {"url": "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth"}
        elif self == Extractor.disk:
            return {
                "input_dim": 128,
                "url": "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/disk_lightglue.pth",
                }
