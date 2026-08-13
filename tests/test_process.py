"""Round-trip tests for the normalization utilities in data/process.py.

transNormbyDic(toNorm=True) followed by transNormbyDic(toNorm=False) should
recover the original values (within tolerance), for both the gamma-log
"special var" path (precipitation-like variables) and the plain path.
"""
import torch
import pytest

from data.process import getStatDic, transNormbyDic


def test_roundtrip_precip_like_variable():
    torch.manual_seed(0)
    # (coords, time, 1 feature) of non-negative precip-like values
    x = torch.rand(5, 100, 1) * 20.0

    stat_dict = getStatDic(flow_regime=0, seriesLst=["pr"], seriesdata=x)
    normed = transNormbyDic(x, ["pr"], stat_dict, toNorm=True, flow_regime=0)
    recovered = transNormbyDic(normed, ["pr"], stat_dict, toNorm=False, flow_regime=0)

    assert torch.allclose(recovered, x, atol=1e-2, rtol=1e-2)


def test_roundtrip_plain_variable():
    torch.manual_seed(1)
    # Not in the special_vars list -> plain (x - mean) / std normalization
    x = torch.randn(4, 50, 1) * 3.0 + 10.0

    stat_dict = getStatDic(flow_regime=0, seriesLst=["elev"], seriesdata=x)
    normed = transNormbyDic(x, ["elev"], stat_dict, toNorm=True, flow_regime=0)
    recovered = transNormbyDic(normed, ["elev"], stat_dict, toNorm=False, flow_regime=0)

    assert torch.allclose(recovered, x, atol=1e-4, rtol=1e-4)


def test_roundtrip_2d_attribute_tensor():
    torch.manual_seed(2)
    # (coords, features) shape, as used for static attributes
    x = torch.rand(10, 2) * 100.0

    stat_dict = getStatDic(flow_regime=0, attrLst=["slope", "aspect"], attrdata=x)
    normed = transNormbyDic(x, ["slope", "aspect"], stat_dict, toNorm=True, flow_regime=0)
    recovered = transNormbyDic(normed, ["slope", "aspect"], stat_dict, toNorm=False, flow_regime=0)

    assert torch.allclose(recovered, x, atol=1e-2, rtol=1e-2)
