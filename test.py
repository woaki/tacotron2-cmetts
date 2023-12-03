import torch
import numpy as np
from tqdm import tqdm

from train import load_model, DataLoader
from utils import TextMelLoader, TextMelCollate
from hparams import create_hparams

hparams = create_hparams()
EMOTIONS = ["angry", "happy", "surprise", "neutral", "sad"]


def embedding_collection():
    # prepare model
    model = load_model(hparams)
    model.load_state_dict(torch.load(hparams.etts_ada_pth)['state_dict'], strict=True)
    _ = model.cuda().eval()

    # prepare dataloader
    valset = TextMelLoader("filelists/esd_english_train_list.txt", hparams)
    collate_fn = TextMelCollate(hparams)
    val_loader = DataLoader(
        valset, sampler=None, num_workers=1, shuffle=False,
        batch_size=1, pin_memory=False, collate_fn=collate_fn
    )

    f1 = open("plotting_function/data/eng_test_embeddings.txt", "w")
    f2 = open("plotting_function/data/eng_test_labels.txt", "w")
    f = open("plotting_function/data/eng_mean_and_std.txt", "w")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            if i % 10 == 0:
                x, y, style_targets = model.parse_batch(batch, hparams)
                y_pred, style_out, alignments = model(x)

                embedding, _ = style_out
                embedding = embedding.cpu().numpy()
                mean = str(embedding.mean())
                # print(mean)
                std = str(embedding.std())
                f.write(mean + " " + std + "\n")

                emotion_id, _, _ = style_targets
                emotion_id = emotion_id.int().cpu().numpy()

                np.savetxt(f1, embedding, delimiter="\t")
                np.savetxt(f2, emotion_id, fmt='%i', delimiter="\t")

    f1.close()
    f2.close()


if __name__ == '__main__':
    embedding_collection()
