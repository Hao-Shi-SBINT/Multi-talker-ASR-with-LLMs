# 安装 safetensors 库，如果还没安装
# pip install safetensors

from safetensors import safe_open

# 替换成你自己的文件路径
file_path = "/lustre/users/shi/toolkits/m_speaker_llm/Multi-talker-ASR-with-LLMs/exp/mode_hybrid-wavlm-Llama-3.2-1B-Instruct-encoder_freeze-decoder_freeze-adater_decoder-ctc-libri2mix_mini/model_unmerge.safetensors"

# 打开 safetensors 文件
with safe_open(file_path, framework="pt") as f:  # framework 可以是 "pt" 或 "tf"
    print("模型中包含的 key：")
    for key in f.keys():
        print(key)
    
    # 可选：查看某个参数的 shape
    example_key = list(f.keys())[0]
    print(f"\n示例参数 '{example_key}' 的 shape：", f.get_tensor(example_key).shape)

