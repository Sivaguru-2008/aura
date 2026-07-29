"""Corpus readers for the Brain Vision Engine.

One reader today. It sits here rather than in
:mod:`aura.backend.foundation.mri.io` on purpose: the foundation layer's readers handle
*clinical interchange formats* that a hospital or a scanner produces, and a
research-challenge redistribution is a different kind of thing with different
guarantees. Mixing them would mean a clinical upload path could, in principle, be
satisfied by a corpus reader.
"""
from .brats_h5 import (
    BratsCorpusIndex,
    BratsH5Reader,
    BratsSubject,
    ChannelVerification,
)

__all__ = ["BratsCorpusIndex", "BratsH5Reader", "BratsSubject", "ChannelVerification"]
