# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker, SiamRPNTrackerForTrain, SiamRPNTrackerUpdate
from pysot.tracker.siammask_tracker import SiamMaskTracker
from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker

TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamRPNTrackerForTrain': SiamRPNTrackerForTrain,
          'SiamRPNTrackerUpdate': SiamRPNTrackerUpdate,
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNLTTracker': SiamRPNLTTracker
         }


# def build_tracker(model):
#     return TRACKS[cfg.TRACK.TYPE](model)

# def build_tracker(model, warpping_model):
#     return TRACKS[cfg.TRACK.TYPE](model, warpping_model)
#
def build_tracker(model, warpping_model, deepmask_model):
    return TRACKS[cfg.TRACK.TYPE](model, warpping_model, deepmask_model)

def build_tracker_for_train(model, warpping_model, deepmask_model, updatenet):
    return TRACKS['SiamRPNTrackerForTrain'](model, warpping_model, deepmask_model, updatenet)

def build_tracker_for_test(model, warpping_model, deepmask_model, updatenet):
    return TRACKS['SiamRPNTrackerUpdate'](model, warpping_model, deepmask_model, updatenet)
