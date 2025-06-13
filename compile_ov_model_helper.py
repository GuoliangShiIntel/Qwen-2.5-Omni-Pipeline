import io
import openvino as ov

core = ov.Core()

def convert_thinker_audio_embedding_to_static_shape(model):
    shapes = {}
    for input in model.inputs:
        input_shape = input.partial_shape
        input_name = input.any_name

        if input_name.startswith("padded_feature"):
            input_shape[0] = 6
            input_shape[1] = 128
            input_shape[2] = 200
        elif input_name == "padded_mask":
            input_shape[0] = 6
            input_shape[1] = 1
            input_shape[2] = 200

        shapes[input] = input_shape

        print(f"input_name: {input_name}")
        print(f"input_shape: {shapes[input]}")

    model.reshape(shapes)

    return model

def convert_thinker_audio_to_static_shape(model):
    shapes = {}
    for input in model.inputs:
        input_shape = input.partial_shape
        input_name = input.any_name

        if input_name.startswith("hidden_states"):
            input_shape[0] = 512
            input_shape[1] = 1280
        elif input_name == "padded_mask_after_cnn":
            input_shape[0] = 6
            input_shape[1] = 100

        shapes[input] = input_shape

        print(f"input_name: {input_name}")
        print(f"input_shape: {shapes[input]}")

    model.reshape(shapes)

    return model

def convert_thinker_audio_state_to_static_shape(model):
    shapes = {}
    for input in model.inputs:
        input_shape = input.partial_shape
        input_name = input.any_name

        if input_name.startswith("each_audio_states"):
            input_shape[0] = 512
            input_shape[1] = 1280

        shapes[input] = input_shape

        print(f"input_name: {input_name}")
        print(f"input_shape: {shapes[input]}")

    model.reshape(shapes)

    return model

def convert_thinker_vision_to_static_shape(model):
    shapes = {}
    for input in model.inputs:
        input_shape = input.partial_shape
        input_name = input.any_name

        if input_name.startswith("hidden_states"):
            input_shape[0] = 3456
            input_shape[1] = 1176

        shapes[input] = input_shape

        print(f"input_name: {input_name}")
        print(f"input_shape: {shapes[input]}")

    model.reshape(shapes)

    return model

def convert_thinker_vision_merger_to_static_shape(model):
    shapes = {}
    for input in model.inputs:
        input_shape = input.partial_shape
        input_name = input.any_name

        if input_name.startswith("hidden_states"):
            input_shape[0] = 2048
            input_shape[1] = 1280
        elif input_name.startswith("attention_mask"):
            input_shape[0] = 1
            input_shape[1] = 2048
            input_shape[2] = 2048
        elif input_name.startswith("window_attention_mask"):
            input_shape[0] = 1
            input_shape[1] = 2048
            input_shape[2] = 2048
        elif input_name.startswith("window_index"):
            input_shape[0] = 512
        elif input_name.startswith("rotary_pos_emb"):
            input_shape[0] = 2048
            input_shape[1] = 40

        shapes[input] = input_shape

        print(f"input_name: {input_name}")
        print(f"input_shape: {shapes[input]}")

    model.reshape(shapes)

    return model

def convert_token2wav_dit_to_static_shape(model):
    shapes = {}
    for input in model.inputs:
        input_shape = input.partial_shape
        input_name = input.any_name

        if input_name.startswith("hidden_states"):
            input_shape[0] = 1
            input_shape[1] = 1024
            input_shape[2] = 80
        elif input_name.startswith("condition_vector"):
            input_shape[0] = 1
            input_shape[1] = 400
            input_shape[2] = 80
        elif input_name.startswith("speaker_embedding"):
            input_shape[0] = 1
            input_shape[1] = 1024
            input_shape[2] = 192
        elif input_name.startswith("quantized_code"):
            input_shape[0] = 1
            input_shape[1] = 512

        shapes[input] = input_shape

        print(f"input_name: {input_name}")
        print(f"input_shape: {shapes[input]}")

    model.reshape(shapes)

    return model

def convert_token2wav_bigvgan_to_static_shape(model):
    shapes = {}
    for input in model.inputs:
        input_shape = input.partial_shape
        input_name = input.any_name

        if input_name.startswith("mel_spectrogram"):
            input_shape[0] = 1
            input_shape[1] = 80
            input_shape[2] = 1024

        shapes[input] = input_shape

        print(f"input_name: {input_name}")
        print(f"input_shape: {shapes[input]}")

    model.reshape(shapes)

    return model

def update_config(config, pair):
    if pair[0] not in config:
        config[pair[0]] = pair[1]

def rename_key(config, old_key, new_key):
    if old_key in config:
        opt_value = config.pop(old_key)
        config[new_key] = opt_value

class KVAxesPosition:
    def __init__(self, batch: int, seq_len: int):
        self.batch = batch
        self.seq_len = seq_len

class KVDesc:
    def __init__(self, max_prompt_len: int, min_response_len: int):
        self.max_prompt_len = max_prompt_len
        self.min_response_len = min_response_len

def update_npu_config(config, kv_pos, kv_desc):
    update_config(config, ("NPU_USE_NPUW", "YES"))
    update_config(config, ("NPUW_LLM", "YES"))
    # update_config(config, ("NPUW_DUMP_SUBS", "MIN"))

    update_config(config, ("NPUW_LLM_BATCH_DIM", kv_pos.batch))
    update_config(config, ("NPUW_LLM_SEQ_LEN_DIM", kv_pos.seq_len))

    update_config(config, ("NPUW_LLM_MAX_PROMPT_LEN", kv_desc.max_prompt_len))
    update_config(config, ("NPUW_LLM_MIN_RESPONSE_LEN", kv_desc.min_response_len))

    update_config(config, ("NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES"))

    rename_key(config, "++PREFILL_CONFIG", "++NPUW_LLM_PREFILL_CONFIG")
    rename_key(config, "++GENERATE_CONFIG", "++NPUW_LLM_GENERATE_CONFIG")
    rename_key(config, "PREFILL_CONFIG", "NPUW_LLM_PREFILL_CONFIG")
    rename_key(config, "PREFILL_HINT", "NPUW_LLM_PREFILL_HINT")
    rename_key(config, "GENERATE_CONFIG", "NPUW_LLM_GENERATE_CONFIG")
    rename_key(config, "GENERATE_HINT", "NPUW_LLM_GENERATE_HINT")

def npu_model_import_or_compile(blob_path, model_path, convert_func, device, model_type, config=None):
    """
    Import or compile blob for NPU device, either audio or vision encoder.

    Parameters:
    - blob_path: Path to the compiled blob file.
    - model_path: Path to the model file.
    - convert_func: Function to convert the model to a static shape.
    - device: Device to compile the model on.
    - model_type: Type of the model
    """

    assert device == "NPU"

    if blob_path.exists():
        try:
            with blob_path.open("rb") as fin:
                print(f"Import {model_type} compiled blob, device: {device}")
                model = core.import_model(fin.read(), device)
                print(f" - Import {model_type} blob done")
        except IOError:
            raise Exception(f"{model_type.capitalize()} blob file can't be opened")
    else:
        model = core.read_model(model_path)
        if model_type == 'thinker_audio_embedding':
            model = convert_func(model)
            ir_name = model_type + "_static.xml"
        elif model_type == 'thinker_audio':
            model = convert_func(model)
            ir_name = model_type + "_static.xml"
        elif model_type == 'thinker_audio_state':
            model = convert_func(model)
            ir_name = model_type + "_static.xml"
        elif model_type == 'thinker_vision':
            model = convert_func(model)
            ir_name = model_type + "_static.xml"
        elif model_type == 'thinker_vision_merger':
            model = convert_func(model)
            ir_name = model_type + "_static.xml"
        elif model_type == 'token2wav_dit':
            model = convert_func(model)
            ir_name = model_type + "_static.xml"
        elif model_type == 'token2wav_bigvgan':
            model = convert_func(model)
            ir_name = model_type + "_static.xml"
        else:
            print(f"Unsupported {model_type}")
            assert(1)

        ov.serialize(model, blob_path.parent / ir_name)
        print(f"Start to compile {ir_name}, device: {device}")
        config = {}
        update_config(config, ("NPU_TILES", "6"))
        model = core.compile_model(model, device, config)
        try:
            user_stream = io.BytesIO()
            model.export_model(user_stream)
            with blob_path.open("wb") as fout:
                fout.write(user_stream.getbuffer())
        except IOError:
            raise Exception(f"{model_type.capitalize()} blob file can't be exported")
        print(f" - Compile {model_type} done")

    return model

def npu_llm_model_import_or_compile(blob_path, model_path, weights_bin, device, model_type):
    """
    Import or compile blob for NPU device, either audio or vision encoder.

    Parameters:
    - blob_path: Path to the compiled blob file.
    - model_path: Path to the model file.
    - convert_func: Function to convert the model to a static shape.
    - device: Device to compile the model on.
    - model_type: Type of the model ('audio' or 'vision').
    """
    config = {}

    assert device == "NPU"

    update_config(config, ("WEIGHTS_PATH", str(weights_bin)))
    if blob_path.exists():
        try:
            with blob_path.open("rb") as fin:
                print(f"Import {model_type} compiled blob, device: {device}")
                model = core.import_model(fin.read(), device, config)
                print(f" - Import llm NPUW blob done")
        except IOError:
            raise Exception(f"blob file can't be opened")
    else:
        print(f"Start to compile {model_type}, device: {device}")
        model = core.read_model(model_path)
        kv_desc = KVDesc(max_prompt_len=1024, min_response_len=128)
        kv_pos = KVAxesPosition(batch=0, seq_len=2)
        update_npu_config(config, kv_pos, kv_desc)
        model = core.compile_model(model, device, config)
        try:
            user_stream = io.BytesIO()
            model.export_model(user_stream)
            with blob_path.open("wb") as fout:
                fout.write(user_stream.getbuffer())
        except IOError:
            raise Exception(f"blob file can't be exported")
        print(f" - Compile llm done")

    return model

def cpu_gpu_model_import_or_compile(blob_path, model_path, device, model_type, blob_cache=True):
    """
    Import or compile blob for CPU or GPU device

    Parameters:
    - blob_path: Path to the compiled blob file.
    - model_path: Path to the model file.
    - device: Device to compile the model on.
    - model_type: Type of the model
    """

    if blob_cache and blob_path.exists():
        try:
            with blob_path.open("rb") as fin:
                print(f"Import {model_type} compiled blob, device: {device}")
                model = core.import_model(fin.read(), device)
                print(f" - Import {model_type} blob done")
        except IOError:
            raise Exception(f"{model_type.capitalize()} blob file can't be opened")
    else:
        model = core.read_model(model_path)
        print(f"Start to compile {model_type}, device: {device}")
        model = core.compile_model(model, device)
        if blob_cache:
            try:
                user_stream = io.BytesIO()
                model.export_model(user_stream)
                with blob_path.open("wb") as fout:
                    fout.write(user_stream.getbuffer())
            except IOError:
                raise Exception(f"{model_type.capitalize()} blob file can't be exported")
        print(f" - Compile {model_type} done")

    return model
