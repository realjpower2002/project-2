
import os

if os.path.exists("spm_data.txt"):
    print("Training data already exists")

else:
    import json
    spm_data = ""

    with open("../data/train.jsonl", "r") as f:
        json_lines = list(f)

        for line in json_lines:
            line_data = json.loads(line)
            spm_data += line_data["prompt"] + " \n"
            spm_data += line_data["completion"] + " \n"
        
    with open("spm_data.txt", "w") as f:
        f.write(spm_data)



if os.path.exists("m.model"):
    print("Model already exists.")
    
else:
    import sentencepiece as spm
    spm.SentencePieceTrainer.train('--input=spm_data.txt --model_type=bpe --model_prefix=m --bos_id=2 --eos_id=3 --bos_piece=<bos> --eos_piece=<eos> --vocab_size=10000')

    sp = spm.SentencePieceProcessor()
    sp.load('m.model')

    # encode: text => id
    print(sp.encode_as_pieces('This is a test'))
    print(sp.encode_as_ids('This is a test'))

    # decode: id => text
    print(sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est']))
    print(sp.decode_ids([519, 86, 5, 4822]))