from model_convert_helper import convert_qwen2_5_omni_model
from data_preprocess_helper import get_wikitext2_dataset

import nncf
from pathlib import Path

# model_id = "Qwen/Qwen2.5-Omni-7B"
model_id = "Qwen/Qwen2.5-Omni-3B"
model_dir = Path(model_id.split("/")[-1])

# compression_configuration = {"mode": nncf.CompressWeightsMode.INT4_SYM,
#                              "group_size": 128, 
#                              "ratio": 1.0, 
#                              "all_layers": False,
#                              "backup_mode": nncf.BackupMode.NONE
#                             }
# model_dir = model_dir.with_name(model_dir.name + "-INT4-SYM")

# Enable dataset-aware quantization for NF4 using wikitext2
use_dataset_aware = True
enable_awq = True
enable_scale_estimation = True

compression_configuration = {"mode": nncf.CompressWeightsMode.NF4,
                             "group_size": -1, 
                             "ratio": 1.0, 
                             "all_layers": False,
                             "backup_mode": nncf.BackupMode.NONE
                            }

# Add AWQ and scale estimation options for better accuracy
if enable_awq:
    compression_configuration["awq"] = True
if enable_scale_estimation:
    compression_configuration["scale_estimation"] = True

model_dir = model_dir.with_name(model_dir.name + "-NF4")

# Get calibration dataset for dataset-aware quantization
calibration_dataset = None
if use_dataset_aware and compression_configuration["mode"] == nncf.CompressWeightsMode.NF4:
    try:
        print("⌛ Loading wikitext2 dataset for calibration...")
        calibration_dataset = get_wikitext2_dataset(model_id)
        print("✅ Wikitext2 dataset loaded successfully")
    except Exception as e:
        print(f"⚠️ Failed to load wikitext2 dataset: {e}")
        print("ℹ️ Proceeding with standard quantization without dataset calibration")
        calibration_dataset = None

convert_qwen2_5_omni_model(model_id, model_dir, compression_configuration, calibration_dataset)

# Error 1:
# Log: requests.exceptions.SSLError: (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): \
#      Max retries exceeded with url: /Qwen/Qwen2.5-Omni-7B/resolve/main/spk_dict.pt (Caused by SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1007)')))")
# Fix: set HF_ENDPOINT=https://hf-mirror.com