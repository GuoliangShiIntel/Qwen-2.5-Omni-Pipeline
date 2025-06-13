from model_convert_helper import convert_qwen2_5_omni_model

import nncf
from pathlib import Path

model_id = "Qwen/Qwen2.5-Omni-7B"
model_dir = Path(model_id.split("/")[-1])

# compression_configuration = {"mode": nncf.CompressWeightsMode.INT4_SYM,
#                              "group_size": 128, 
#                              "ratio": 1.0, 
#                              "all_layers": False,
#                              "backup_mode": nncf.BackupMode.NONE
#                             }
# model_dir = model_dir.with_name(model_dir.name + "-INT4-SYM")

compression_configuration = {"mode": nncf.CompressWeightsMode.NF4,
                             "group_size": -1, 
                             "ratio": 1.0, 
                             "all_layers": False,
                             "backup_mode": nncf.BackupMode.NONE
                            }
model_dir = model_dir.with_name(model_dir.name + "-NF4")

convert_qwen2_5_omni_model(model_id, model_dir, compression_configuration)

# Error 1:
# Log: requests.exceptions.SSLError: (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): \
#      Max retries exceeded with url: /Qwen/Qwen2.5-Omni-7B/resolve/main/spk_dict.pt (Caused by SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1007)')))")
# Fix: set HF_ENDPOINT=https://hf-mirror.com