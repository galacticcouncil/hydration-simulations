"""
These settings should be updated to match the production environment.
"""
from hydradx.model.amm.omnipool_amm import DynamicFee
# fee defaults

omnipool_asset_fee_minimum = 0.0025
omnipool_asset_fee_maximum = 0.05
omnipool_asset_fee_amplification = 2.0
omnipool_asset_fee_decay = 0.00001

omnipool_lrna_fee_minimum = 0.0005
omnipool_lrna_fee_maximum = 0.001
omnipool_lrna_fee_amplification = 1.0
omnipool_lrna_fee_decay = 0.000005

omnipool_asset_fee = DynamicFee(
    minimum=omnipool_asset_fee_minimum,
    maximum=omnipool_asset_fee_maximum,
    amplification=omnipool_asset_fee_amplification,
    decay=omnipool_asset_fee_decay
)
omnipool_lrna_fee = DynamicFee(
    minimum=omnipool_lrna_fee_minimum,
    maximum=omnipool_lrna_fee_maximum,
    amplification=omnipool_lrna_fee_amplification,
    decay=omnipool_lrna_fee_decay
)