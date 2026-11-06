# Tests check the code's default rules, not whatever settings the nightly tuning adopted.
import os
os.environ["USE_TUNED_PARAMS"] = "false"
