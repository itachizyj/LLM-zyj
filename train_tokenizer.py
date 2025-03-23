import random
import json
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)

import os

def train_tokenizer():
    def read_texts_from_jsonl(file_path):
        with open(file_path, "r", encoding = "utf-8") as f:
            for line in f:
                data = json.loads(line)
                yield data["text"]
    data_path = "pretrain.jsonl"

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)

    special_tokens = ["<unk>", "<s>", "</s>"]

    trainer = trainers.BpeTrainer(
        vocab_size = 6400,
        special_tokens = special_tokens,
        show_progress = True,
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
    )
    texts = read_texts_from_jsonl(data_path)

    tokenizer.train_from_iterator(texts, trainer = trainer)

    tokenizer.decoder = decoders.ByteLevel()
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2

    tokenizer_dir = "./zyj_tokenizer"
    os.makedirs(tokenizer_dir, exist_ok = True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("./zyj_tokenizer")

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder":{
            "0":{
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1":{
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2":{
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "addtional_special tokens": [],
        #额外的特殊符号(为空)
        "bos_token":"<s>",
        #开始符号
        "eos token":"</s>",
        #结束符号
        "clean up tokenization_spaces" : False,
        #不清理空格
        "legacy" : True,
        #兼容旧版
        "model max length" : 32768,
        #模型最大长度
        "pad token" : "<unk>",
        #填充符号
        "sp_model_kwargs" : {},
        #子词模型的其他参数
        "spaces_between_special_tokens" : False,
        #不在特铁符号之间添加空格能 
        "tokenizer class":"PreTrainedTokenizerFast",
        #使用的分词器类型
        "unk token":"<unk>",
        #未知符号的token
        "chat_template": "{% if messages[0]['role'] == 'system' %} {% set system_message = messages[0]['content'] %} {{'<s>systeml\\n' + system_message +'</s\\n' }}{% else %}{{ '<s>system\\n 你是 zyj_llm，是一个有用的人工智能助手。</s>\\n' }} {% endif %} {% for message in messages %} {% set content = message['content'] %}{% if message['role'] == 'user' %} {{'<s>user\\n'+ content +'</s>\\n<s>assistant\\n'}} {% elif message['role'] == 'assistant’%} {{ content +'</s>'+\\n'}}{% endif %}{% endfor %}"
    }



    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    train_tokenizer()



